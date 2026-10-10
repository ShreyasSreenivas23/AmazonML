"""Stage-2 LightGBM tuning on a cached frame (E026): same split, groups and decision rule as src.er_fullpass stage2.

    python scripts/tune_stage2.py --frame runs/frames/E025/train_frame.parquet --out runs/E026-tune

Each variant: 4 internal S1-grouped OOF models on the fit pool, decision rule chosen on OOF, one dev F0.5. Then a
seed ensemble (mean probability of the best variant over 3 seeds). Results append to <out>/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir, load  # noqa: E402
from src.er_model import predict, train_oof  # noqa: E402
from src.er_pipeline import apply_decision, choose_decision  # noqa: E402
from src.metrics import er_fbeta_macro  # noqa: E402

VARIANTS = {
    "base_lr0.1": {},
    "lr0.05": {"learning_rate": 0.05, "_rounds": 3000},
    "leaves255": {"num_leaves": 255},
    "leaves63": {"num_leaves": 63},
    "minleaf300": {"min_data_in_leaf": 300},
    "ff0.6": {"feature_fraction": 0.6},
    "l2_10": {"lambda_l2": 10.0},
    "lr0.05_leaves255": {"learning_rate": 0.05, "num_leaves": 255, "_rounds": 3000},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _, _, _, gt = load("train")
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev = set(folds.loc[folds.dev, "s1_id"])

    def truth(ids):
        ids = set(ids)
        g = gt[gt.s1_id.isin(ids)]
        y = {s: set() for s in ids}
        for s, c in zip(g.s1_id, g.cand_id):
            y[s].add(c)
        return y

    def prep(X):
        f = X.drop(columns=["prob"])
        f["s1_prob"] = X.prob.to_numpy()
        return f

    X = pd.read_parquet(a.frame)
    D = X[X.s1_id.isin(dev)].reset_index(drop=True)
    T = X[~X.s1_id.isin(dev)].reset_index(drop=True)
    del X
    y_dev, y_tr = truth(dev), truth(T.s1_id.unique())
    fT, fD = prep(T), prep(D)
    ids = pd.Index(T.s1_id.unique())

    def run(params: dict, seed: int = 0):
        p = {k: v for k, v in params.items() if not k.startswith("_")}
        p["seed"] = 42 + seed
        g = pd.Series(np.random.default_rng(seed).integers(0, 4, len(ids)), index=ids).reindex(T.s1_id).to_numpy()
        oof, models = train_oof(fT, T.y.to_numpy().astype(int), g, p, num_boost_round=params.get("_rounds", 1000),
                                early_stopping=50)
        return oof, predict(models, fD)

    def score(oof, pdev):
        rule, t, f_oof = choose_decision(T[["s1_id", "cand_id"]].assign(prob=oof), y_tr)
        dev_f = er_fbeta_macro(y_dev, apply_decision(D[["s1_id", "cand_id"]].assign(prob=pdev), rule, t))
        return rule, t, f_oof, dev_f

    results = {}
    for name, params in VARIANTS.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        oof, pdev = run({"learning_rate": 0.1, **params})
        rule, t, f_oof, dev_f = score(oof, pdev)
        results[name] = (dev_f, params)
        row = {"variant": name, "oof": f_oof, "dev": dev_f, "rule": rule, "t": t, "s": round(time.time() - t0)}
        print(json.dumps(row), flush=True)
        with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    best = max(results, key=lambda k: results[k][0])
    params = {"learning_rate": 0.1, **results[best][1]}
    oofs, pdevs = [], []
    for sd in range(a.seeds):
        oof, pdev = run(params, seed=sd)
        oofs.append(oof)
        pdevs.append(pdev)
    rule, t, f_oof, dev_f = score(np.mean(oofs, 0), np.mean(pdevs, 0))
    row = {"variant": f"{best}_x{a.seeds}seeds", "oof": f_oof, "dev": dev_f, "rule": rule, "t": t}
    print(json.dumps(row), flush=True)
    with open(out / "results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
