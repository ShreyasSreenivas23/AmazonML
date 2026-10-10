"""Merge two cross-encoders into leak-free per-chunk scores for stage 2 (two-fold cross-encoder, E011).

CE A trained on S1 folds 1-2, CE B on folds 3-4. For every train S1 we take the score from the model that did NOT
train on it; dev (fold 0) and test get the mean of both models.

    python scripts/merge_ce.py --a runs/E008-ce --a-extra runs/E008-ce/train_ce_f4 --b runs/E011-ceB --out runs/E011-ce2

Inputs (src.er_crossenc score outputs, one parquet per stage-1 chunk, same row order as the chunk):
    A: <a>/train_ce (folds 0,3 scored), <a-extra> (fold 4 scored), <a>/test_ce
    B: <b>/train_ce (folds 0,1,2 scored), <b>/test_ce
Output: <out>/train_ce/*.parquet and <out>/test_ce/*.parquet with s1_id, cand_id, ce_score (no NaN for folds 0-4).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402

TRAINED_ON = {"A": {1, 2}, "B": {3, 4}}


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["s1_id", "cand_id", "ce_score"])


def check_aligned(*frames: pd.DataFrame) -> None:
    base = frames[0]
    for f in frames[1:]:
        if len(f) != len(base) or not (f.s1_id.to_numpy() == base.s1_id.to_numpy()).all() \
                or not (f.cand_id.to_numpy() == base.cand_id.to_numpy()).all():
            raise ValueError("CE score files are not row-aligned")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--a-extra", required=True, help="A's scores for fold 4 (a separate score run)")
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").fold
    out = Path(a.out)
    (out / "train_ce").mkdir(parents=True, exist_ok=True)
    (out / "test_ce").mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in (Path(a.b) / "train_ce").glob("*.parquet"))
    for name in names:
        fa, fa4, fb = read(Path(a.a) / "train_ce" / name), read(Path(a.a_extra) / name), read(Path(a.b) / "train_ce" / name)
        check_aligned(fa, fa4, fb)
        f = folds.reindex(fa.s1_id).to_numpy()
        sa = np.where(f == 4, fa4.ce_score.to_numpy(), fa.ce_score.to_numpy())
        sb = fb.ce_score.to_numpy()
        score = np.select([np.isin(f, list(TRAINED_ON["A"])), np.isin(f, list(TRAINED_ON["B"])), f == 0],
                          [sb, sa, (sa + sb) / 2], default=np.nan).astype(np.float32)
        if np.isnan(score).any():
            raise ValueError(f"{name}: {int(np.isnan(score).sum())} rows without an out-of-sample CE score")
        fa.assign(ce_score=score).to_parquet(out / "train_ce" / name, index=False)
    for name in sorted(p.name for p in (Path(a.a) / "test_ce").glob("*.parquet")):
        ta, tb = read(Path(a.a) / "test_ce" / name), read(Path(a.b) / "test_ce" / name)
        check_aligned(ta, tb)
        ta.assign(ce_score=((ta.ce_score.to_numpy() + tb.ce_score.to_numpy()) / 2).astype(np.float32)) \
            .to_parquet(out / "test_ce" / name, index=False)
    print(f"merged {len(names)} train chunks and test chunks into {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
