"""Inputs of `src.er_owner prep` from the stage-1 chunks (no stage-2 frames needed):
  <out>/pairs.parquet  stage-1 pairs of every train S1 (s1_id, cand_id, prob): training groups
  <out>/fold0.parquet  the fold-0 subset (the stage-2 fit pool + dev), whose records get out-of-sample owner scores
  <out>/test.parquet   every test pair

    python scripts/owner_pairs.py --out runs/OW04
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402


def read(chunks: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "prob"]) for f in sorted(chunks.glob("*.parquet"))],
                     ignore_index=True)


def main() -> None:
    """Write pairs.parquet, fold0.parquet and test.parquet for the owner-model prep step."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-root", default="runs/E015")
    ap.add_argument("--out", default="runs/OW04")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = read(Path(a.chunks_root) / "train_chunks")
    tr.to_parquet(out / "pairs.parquet", index=False)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    tr[tr.s1_id.isin(set(folds.loc[folds.fold == 0, "s1_id"]))].to_parquet(out / "fold0.parquet", index=False)
    read(Path(a.chunks_root) / "test_chunks").to_parquet(out / "test.parquet", index=False)
    print(f"wrote {out}/pairs.parquet, fold0.parquet, test.parquet")


if __name__ == "__main__":
    main()
