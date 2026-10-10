"""E028: covariate-shift (importance-weighted) stage 2 for S1s of countries unseen in train. Label-free, country-agnostic.

Why: on test, S1s of the unseen country face a much denser competition regime than train (27 Sep: competing S1s per
plausible candidate record 4.16 vs 2.75-2.96; best competing name cosine 0.89 vs 0.80-0.82). Stage 2 learned on
US/India that competition means "distractor", so it under-matches there (3.256 vs 3.39 matches/S1).
How: a domain classifier on competition-only features (train fit rows vs unseen-country test rows) gives importance
weights w = p/(1-p); stage 2 is refitted with them (same frames, features, internal S1 groups, LightGBM settings as
er_fullpass stage2) and used ONLY for unseen-country S1s; seen countries keep the base run's probabilities.
Candidate rows are unchanged (same cached, pruned frames) -> candidate_pairs.tsv byte-identical to the base run.

    python scripts/unseen_weighted.py --frames runs/frames/E023b_full --base runs/E015/sub_E023b_full \
        --tag E023b_full --out runs/E015/sub_E028_uw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir  # noqa: E402
from src.er_decide import decide  # noqa: E402
from src.er_fullpass import prune_rows  # noqa: E402
from src.er_model import DEFAULT_PARAMS, feature_cols, predict  # noqa: E402
from src.metrics import er_fbeta_macro  # noqa: E402

DOMAIN_COLS = ["ctx_rank_in_s1", "ctx_n_s1_for_cand", "s2_rank_in_s1", "s2_other_s1_best", "s2_margin_vs_other_s1",
               "comp_cos_name_other_best", "comp_cos_name_margin", "comp_name_ratio_other_best", "comp_name_ratio_margin",
               "comp_name_jw_other_best", "comp_name_jw_margin", "comp_name_full_tset_other_best",
               "comp_name_full_tset_margin", "comp_name_tsort_other_best", "comp_name_tsort_margin", "n_cands"]


def fit_weighted(feat: pd.DataFrame, y: np.ndarray, w: np.ndarray, groups: np.ndarray, lr: float):
    """er_model.train_oof with sample weights (early stopping on unweighted validation logloss)."""
    params = {**DEFAULT_PARAMS, "learning_rate": lr}
    cols = feature_cols(feat)
    oof, models = np.full(len(feat), np.nan), []
    for g in np.unique(groups):
        va = groups == g
        dtr = lgb.Dataset(feat.loc[~va, cols], label=y[~va], weight=w[~va], free_raw_data=True)
        dva = lgb.Dataset(feat.loc[va, cols], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, 1000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(feat.loc[va, cols], num_iteration=m.best_iteration)
        models.append(m)
    return oof, models


def with_n_cands(X: pd.DataFrame) -> pd.DataFrame:
    return X.assign(n_cands=X.groupby("s1_id").cand_id.transform("size").astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--base", required=True, help="dir with the base run's test_probs_stage2.parquet")
    ap.add_argument("--tag", required=True, help="base stage-2 tag (runs/E015/dev_stage2_<tag>.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=20.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    c = cache_dir()
    folds = pd.read_parquet(c / "folds_s1_k5.parquet")
    dev_ids = set(folds.loc[folds.dev, "s1_id"])
    train_countries = set(pd.read_parquet(c / "train_s1.parquet", columns=["country"]).country.unique())
    ts1 = pd.read_parquet(c / "test_s1.parquet", columns=["entity_id", "country"])
    unseen_ids = ts1.loc[~ts1.country.isin(train_countries), "entity_id"].tolist()
    print(f"unseen-country test S1s: {len(unseen_ids):,} ({sorted(set(ts1.country) - train_countries)})", flush=True)

    X = prune_rows(pd.read_parquet(Path(a.frames) / "train_frame.parquet"), "train", 0.2, "runs/E016-ce", 0.01)
    X = with_n_cands(X)
    is_dev = X.s1_id.isin(dev_ids).to_numpy()
    Xtr, Xdev = X[~is_dev].reset_index(drop=True), X[is_dev].reset_index(drop=True)
    del X
    Xt = ds.dataset(Path(a.frames) / "test_frame.parquet").to_table(filter=ds.field("s1_id").isin(unseen_ids)).to_pandas()
    Xt = with_n_cands(prune_rows(Xt, "test", 0.2, "runs/E016-ce", 0.01))
    print(f"train fit rows {len(Xtr):,} | dev rows {len(Xdev):,} | unseen test rows {len(Xt):,}", flush=True)

    # domain classifier: competition-only features, train fit rows (0) vs unseen-country test rows (1)
    dcols = [k for k in DOMAIN_COLS if k in Xtr.columns]
    rng = np.random.default_rng(0)
    si = rng.choice(len(Xtr), min(600_000, len(Xtr)), replace=False)
    ti = rng.choice(len(Xt), min(600_000, len(Xt)), replace=False)
    D = pd.concat([Xtr.loc[si, dcols], Xt.loc[ti, dcols]], ignore_index=True)
    dy = np.r_[np.zeros(len(si)), np.ones(len(ti))]
    perm = rng.permutation(len(D))
    cut = int(0.8 * len(D))
    dtr = lgb.Dataset(D.iloc[perm[:cut]], label=dy[perm[:cut]])
    dva = lgb.Dataset(D.iloc[perm[cut:]], label=dy[perm[cut:]], reference=dtr)
    dm = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "min_data_in_leaf": 500,
                    "verbose": -1, "num_threads": 32, "seed": 0}, dtr, 300, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(dy[perm[cut:]], dm.predict(D.iloc[perm[cut:]], num_iteration=dm.best_iteration))
    p = np.clip(dm.predict(Xtr[dcols], num_iteration=dm.best_iteration), 1e-4, 1 - 1e-4)
    w = p / (1 - p)
    w = np.clip(w / w.mean(), 1 / a.clip, a.clip)
    w = w / w.mean()
    ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
    print(f"domain AUC {auc:.3f} | weights: median {np.median(w):.3f} p99 {np.quantile(w, .99):.2f} max {w.max():.2f} | ESS {ess:.3f}", flush=True)

    # weighted stage 2 (same features as er_fullpass: prob -> s1_prob; + n_cands is NOT a model feature)
    def feats(Z: pd.DataFrame) -> pd.DataFrame:
        f = Z.drop(columns=["prob", "n_cands"])
        f["s1_prob"] = Z.prob.to_numpy()
        return f
    ids = pd.Index(Xtr.s1_id.unique())
    grp = pd.Series(np.random.default_rng(0).integers(0, 4, len(ids)), index=ids).reindex(Xtr.s1_id).to_numpy()
    y = Xtr.y.to_numpy().astype(int)
    oof, models = fit_weighted(feats(Xtr), y, w, grp, a.lr)

    # dev check (US/India labels): overall and on the France-like (high domain score) S1s, base vs weighted
    gt = pd.read_parquet(c / "train_gt_pairs.parquet", columns=["s1_id", "cand_id"])
    gt = gt[gt.s1_id.isin(dev_ids)]
    ytrue = {s: set() for s in dev_ids}
    for s, cnd in zip(gt.s1_id, gt.cand_id):
        ytrue[s].add(cnd)
    pw = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models, feats(Xdev)))
    pb = pd.read_parquet(f"runs/E015/dev_stage2_{a.tag}.parquet", columns=["s1_id", "cand_id", "prob"])
    dscore = pd.Series(dm.predict(Xdev[dcols], num_iteration=dm.best_iteration)).groupby(Xdev.s1_id.to_numpy()).mean()
    hi = set(dscore.index[dscore >= dscore.quantile(0.8)])
    res = {"domain_auc": round(auc, 4), "ess": round(ess, 4)}
    for t in (0.6, 0.7, 0.75, 0.8):
        for name, pp in (("base", pb), ("weighted", pw)):
            m = decide(pp, t, assign=True)
            res[f"{name}_t{t}_dev"] = round(er_fbeta_macro(ytrue, m), 5)
            res[f"{name}_t{t}_dev_hi20"] = round(er_fbeta_macro({k: ytrue[k] for k in hi}, m), 5)
    print(json.dumps(res), flush=True)

    # test: weighted probs for unseen-country S1s, base probs elsewhere (same rows)
    pt = Xt[["s1_id", "cand_id"]].assign(prob=predict(models, feats(Xt)).astype(np.float32))
    base = pd.read_parquet(Path(a.base) / "test_probs_stage2.parquet")
    uns = base.s1_id.isin(set(unseen_ids)).to_numpy()
    key = base[uns][["s1_id", "cand_id"]].merge(pt, on=["s1_id", "cand_id"], how="left")
    if len(key) != int(uns.sum()) or key.prob.isna().any():
        raise ValueError(f"unseen rows misaligned: base {int(uns.sum()):,} vs weighted {len(pt):,}, missing {int(key.prob.isna().sum())}")
    newp = base.copy()
    newp.loc[uns, "prob"] = key.prob.to_numpy()
    newp.to_parquet(out / "test_probs_stage2.parquet", index=False)
    n_fr = len(unseen_ids)
    for t in (0.6, 0.7, 0.75, 0.8):
        k = sum(len(v) for v in decide(newp[uns], t, assign=True).values())
        kb = sum(len(v) for v in decide(base[uns], t, assign=True).values())
        res[f"unseen_mean_matches_t{t}"] = [round(kb / n_fr, 3), round(k / n_fr, 3)]
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("unseen mean matches [base, weighted]:", {k: v for k, v in res.items() if k.startswith("unseen_mean")}, flush=True)


if __name__ == "__main__":
    main()
