"""Convert listwise owner-model scores (src/er_owner.py score -> <dir>/<split>_owner.parquet with s1_id, cand_id, own_p,
own_margin, ...) into the stage-2 --ce-dir format: one parquet per stage-1 chunk, row-aligned with
runs/E015/<split>_chunks, columns s1_id, cand_id, ce_score (NaN where the record was not contested / not scored).

    python scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_p      --out runs/OW04p
    python scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_margin --out runs/OW04m
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Write <out>/{train_ce,test_ce}/NNN.parquet aligned with the stage-1 chunks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-dir", required=True, help="dir with train_owner.parquet and test_owner.parquet")
    ap.add_argument("--col", required=True, choices=["own_p", "own_margin"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunks-root", default="runs/E015")
    a = ap.parse_args()
    for split in ("train", "test"):
        own = pd.read_parquet(Path(a.owner_dir) / f"{split}_owner.parquet", columns=["s1_id", "cand_id", a.col])
        own = own.drop_duplicates(["s1_id", "cand_id"])
        out = Path(a.out) / f"{split}_ce"
        out.mkdir(parents=True, exist_ok=True)
        n = scored = 0
        for chunk in sorted((Path(a.chunks_root) / f"{split}_chunks").glob("*.parquet")):
            c = pd.read_parquet(chunk, columns=["s1_id", "cand_id"])
            m = c.merge(own, on=["s1_id", "cand_id"], how="left")          # left merge keeps the chunk's row order
            c["ce_score"] = m[a.col].to_numpy(np.float32)
            c.to_parquet(out / chunk.name, index=False)
            n += len(c)
            scored += int(c.ce_score.notna().sum())
        print(f"{split}: {n:,} rows, scored {scored / max(n, 1):.4f} -> {out}")


if __name__ == "__main__":
    main()
