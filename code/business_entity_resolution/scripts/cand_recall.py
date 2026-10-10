"""Candidate-file metrics of one stage-2 run (targets: ~4.7 cands/S1, recall >= 0.996 on dev AND fit pool, reduction
ratio >= 99.9998%):
  dev   : candidates/S1 and candidate recall (share of ALL dev ground-truth pairs that survive retrieval + the filter),
          from runs/<exp>/dev_stage2_<tag>.parquet;
  fit   : the same on the fit pool (non-dev S1s of the cached train frame, filter re-applied as in prune_rows);
  test  : candidates/S1 and reduction ratio 1 - pairs / (|S1| * (|S2| + |S3|)) from candidate_pairs.tsv.

    python scripts/cand_recall.py --tag E023a_core --sub submissions/sub_E023a_core --frames runs/frames/E023a_core
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402


def recall(pairs: pd.DataFrame, gt: pd.DataFrame, ids: set) -> tuple[float, float]:
    """(candidates per S1, share of the ids' ground-truth pairs present in pairs)."""
    g = gt[gt.s1_id.isin(ids)]
    hit = g.merge(pairs[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="inner")
    return round(len(pairs) / len(ids), 3), round(len(hit) / len(g), 5)


def main() -> None:
    """Print one JSON line with the candidate-file metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="stage-2 --tag (reads runs/<exp>/dev_stage2_<tag>.parquet)")
    ap.add_argument("--exp-dir", default="runs/E015")
    ap.add_argument("--sub", default="", help="submission dir with candidate_pairs.tsv (test cands/S1, reduction ratio)")
    ap.add_argument("--frames", default="", help="stage-2 --frames dir (fit-pool recall from its train_frame.parquet)")
    ap.add_argument("--prune-eps", type=float, default=0.2)
    ap.add_argument("--prune-ce-dir", default="runs/E016-ce")
    ap.add_argument("--prune-ce", type=float, default=0.01)
    a = ap.parse_args()
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev_ids = set(folds.loc[folds.dev, "s1_id"])
    gt = pd.read_parquet(cache_dir() / "train_gt_pairs.parquet", columns=["s1_id", "cand_id"])
    dev = pd.read_parquet(Path(a.exp_dir) / f"dev_stage2_{a.tag}.parquet", columns=["s1_id", "cand_id"])
    out: dict = {"tag": a.tag}
    out["dev_cands_per_s1"], out["dev_cand_recall"] = recall(dev, gt, dev_ids)
    if a.frames:
        X = pd.read_parquet(Path(a.frames) / "train_frame.parquet", columns=["s1_id", "cand_id", "prob"])
        X = X[~X.s1_id.isin(dev_ids)]
        fit_ids = set(X.s1_id.unique())
        keep = X.prob.to_numpy() >= a.prune_eps
        if a.prune_ce_dir:
            ce = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "ce_score"])
                            for f in sorted((Path(a.prune_ce_dir) / "train_ce").glob("*.parquet"))], ignore_index=True)
            v = X[["s1_id", "cand_id"]].merge(ce, on=["s1_id", "cand_id"], how="left").ce_score.to_numpy()
            keep |= np.nan_to_num(v, nan=0.0) >= a.prune_ce
        out["fit_cands_per_s1"], out["fit_cand_recall"] = recall(X[keep], gt, fit_ids)
    if a.sub:
        # one row per S1, candidate ids comma-joined (empty for S1s with no candidates)
        c = pd.read_csv(Path(a.sub) / "candidate_pairs.tsv", sep="\t", dtype=str, keep_default_na=False)
        n = int(c.iloc[:, 1].map(lambda s: len(s.split(",")) if s else 0).sum())
        pool = sum(pq.ParquetFile(cache_dir() / f"test_s{k}.parquet").metadata.num_rows for k in (2, 3))
        out.update({"test_s1": len(c), "test_cands_per_s1": round(n / len(c), 3),
                    "reduction_ratio_pct": round(100 * (1 - n / (len(c) * pool)), 6)})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
