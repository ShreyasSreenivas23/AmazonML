"""Two-fold cross-encoder merge: each fold takes the score of the model that did NOT train on it."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _write(d: Path, name: str, s1, cand, score):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"s1_id": s1, "cand_id": cand, "ce_score": np.asarray(score, np.float32)}).to_parquet(d / name, index=False)


def test_merge_picks_out_of_sample_model(tmp_path):
    data = tmp_path / "data"
    (data / "cache").mkdir(parents=True)
    s1 = [f"S1-{i}" for i in range(5)]
    pd.DataFrame({"s1_id": s1, "fold": [0, 1, 2, 3, 4]}).to_parquet(data / "cache" / "folds_s1_k5.parquet")
    cand = [f"S2-{i}" for i in range(5)]
    nan = np.nan
    # A (trained folds 1-2): scored folds 0,3 in train_ce, fold 4 in the extra run
    _write(tmp_path / "A" / "train_ce", "000.parquet", s1, cand, [0.10, nan, nan, 0.30, nan])
    _write(tmp_path / "A4", "000.parquet", s1, cand, [nan, nan, nan, nan, 0.40])
    _write(tmp_path / "A" / "test_ce", "000.parquet", ["S1-t"], ["S3-t"], [0.20])
    # B (trained folds 3-4): scored folds 0,1,2
    _write(tmp_path / "B" / "train_ce", "000.parquet", s1, cand, [0.50, 0.61, 0.62, nan, nan])
    _write(tmp_path / "B" / "test_ce", "000.parquet", ["S1-t"], ["S3-t"], [0.60])
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_ce.py"), "--a", str(tmp_path / "A"),
                        "--a-extra", str(tmp_path / "A4"), "--b", str(tmp_path / "B"), "--out", str(out)],
                       cwd=ROOT, env={**__import__("os").environ, "DATA_DIR": str(data)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = pd.read_parquet(out / "train_ce" / "000.parquet")
    # fold0 -> mean(A,B); folds 1,2 -> B; fold 3 -> A; fold 4 -> A (extra run)
    np.testing.assert_allclose(m.ce_score, [0.30, 0.61, 0.62, 0.30, 0.40], rtol=1e-6)
    assert list(m.s1_id) == s1 and list(m.cand_id) == cand
    t = pd.read_parquet(out / "test_ce" / "000.parquet")
    np.testing.assert_allclose(t.ce_score, [0.40], rtol=1e-6)


def test_merge_refuses_missing_scores(tmp_path):
    data = tmp_path / "data"
    (data / "cache").mkdir(parents=True)
    pd.DataFrame({"s1_id": ["S1-0"], "fold": [1]}).to_parquet(data / "cache" / "folds_s1_k5.parquet")
    for d in ("A/train_ce", "A4", "B/train_ce"):
        _write(tmp_path / d, "000.parquet", ["S1-0"], ["S2-0"], [np.nan])   # fold 1 needs B's score, which is NaN
    (tmp_path / "A" / "test_ce").mkdir(parents=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "merge_ce.py"), "--a", str(tmp_path / "A"),
                        "--a-extra", str(tmp_path / "A4"), "--b", str(tmp_path / "B"), "--out", str(tmp_path / "o")],
                       cwd=ROOT, env={**__import__("os").environ, "DATA_DIR": str(data)}, capture_output=True, text=True)
    assert r.returncode != 0 and "without an out-of-sample CE score" in (r.stderr + r.stdout)
