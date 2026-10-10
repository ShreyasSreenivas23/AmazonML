"""Recall gain on dev from an extra retrieval view, on top of the current stage-1 candidates (E007 top-15).

    python scripts/recall_gain.py --view both --tag ft --k 10
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_blocking import dense_candidates  # noqa: E402
from src.er_data import cache_dir, load  # noqa: E402
from src.er_embed import emb_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="both")
    ap.add_argument("--tag", default="ft")
    ap.add_argument("--k", type=int, nargs="+", default=[10])
    ap.add_argument("--base", default="runs/E007-fullpass/train_chunks")
    a = ap.parse_args()
    s1, s2, s3, gt = load("train")
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev = set(folds.loc[folds.dev, "s1_id"])
    mask = s1.entity_id.isin(dev).to_numpy()
    q = s1[mask].reset_index(drop=True)
    g = gt[gt.s1_id.isin(dev)]
    truth = set(zip(g.s1_id, g.cand_id))
    base = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id"], filters=[("s1_id", "in", list(dev))])
                      for f in sorted(glob.glob(f"{a.base}/*.parquet"))])
    have = set(zip(base.s1_id, base.cand_id))
    miss = truth - have
    e1 = np.ascontiguousarray(np.load(emb_dir() / f"train_s1_{a.view}_{a.tag}.npy", mmap_mode="r")[mask])
    for K in a.k:
        got = set()
        for k, pool in ((2, s2), (3, s3)):
            ep = np.load(emb_dir() / f"train_s{k}_{a.view}_{a.tag}.npy", mmap_mode="r")
            c = dense_candidates(q, pool.reset_index(drop=True), e1, np.asarray(ep), k=K, view=a.view)
            got |= set(zip(c.s1_id, c.cand_id))
        alone = len(truth & got) / len(truth)
        union = 1 - len(miss - got) / len(truth)
        print(f"{a.view}_{a.tag} top-{K}/source: alone recall {alone:.4f} | union recall {1 - len(miss) / len(truth):.4f}"
              f" -> {union:.4f} | recovers {len(miss & got) / len(miss):.1%} of missed | new pairs/S1 "
              f"{len(got - have) / len(q):.1f}", flush=True)


if __name__ == "__main__":
    main()
