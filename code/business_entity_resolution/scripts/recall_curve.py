"""Blocking recall curve on the dev subset (100k S1 from fold 0) against the FULL train S2/S3 pool.

    python scripts/recall_curve.py --views name addr both --k 50

For every embedding view and for their union: pair recall@K per source and overall, plus candidates per S1.
Writes runs/E002-recall/recall.md and caches candidates to data/cache/cands_dev_<model>.parquet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths  # noqa: E402
from src.er_blocking import blocking_recall, dense_candidates  # noqa: E402
from src.er_data import cache_dir, load  # noqa: E402
from src.er_embed import emb_dir  # noqa: E402

KS = (1, 2, 3, 5, 10, 20, 30, 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", default=["name", "addr", "both"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--k", type=int, default=50)
    a = ap.parse_args()
    t0 = time.time()
    s1, s2, s3, gt = load("train")
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev_ids = folds.loc[folds.dev, "s1_id"]
    dev_mask = s1.entity_id.isin(dev_ids).to_numpy()
    q = s1[dev_mask].reset_index(drop=True)
    rows, all_c = [], []
    for view in a.views:
        e1 = np.load(emb_dir() / f"train_s1_{view}_{a.model}.npy", mmap_mode="r")[dev_mask]
        for src, pool in (("S2", s2), ("S3", s3)):
            ep = np.load(emb_dir() / f"train_s{src[1]}_{view}_{a.model}.npy", mmap_mode="r")
            t = time.time()
            c = dense_candidates(q, pool, np.ascontiguousarray(e1), np.asarray(ep), k=a.k, view=view)
            c["src"] = src
            all_c.append(c)
            r = blocking_recall(c, gt[gt.cand_id.str.startswith(src)], q.entity_id, KS)
            rows.append({"view": view, "src": src, "secs": round(time.time() - t, 1), **r})
            print(f"{view}/{src}: recall@10 {r['recall@10']:.4f} recall@50 {r['recall@50']:.4f} ({time.time() - t:.0f}s)", flush=True)
    cands = pd.concat(all_c, ignore_index=True)
    cands.to_parquet(cache_dir() / f"cands_dev_{a.model}.parquet", index=False)
    # union across views: a pair's rank = best rank over views, per source
    for src in ("S2", "S3"):
        cs = cands[cands.src == src]
        r = blocking_recall(cs, gt[gt.cand_id.str.startswith(src)], q.entity_id, KS)
        rows.append({"view": "UNION", "src": src, "secs": 0, **r})
    r = blocking_recall(cands, gt, q.entity_id, KS)
    rows.append({"view": "UNION", "src": "both", "secs": 0, **r})
    res = pd.DataFrame(rows)
    out = get_paths().runs / "E002-recall"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["view", "src", "secs", "n_true_pairs"] + [f"recall@{k}" for k in KS] + [f"cands@{k}" for k in (5, 10, 20, 50)]
    text = (f"# Blocking recall on dev ({len(q):,} S1, full train pool), model={a.model}\n\n"
            + res[cols].to_markdown(index=False, floatfmt=".4f") + f"\n\n_total {time.time() - t0:.0f}s_\n")
    (out / "recall.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
