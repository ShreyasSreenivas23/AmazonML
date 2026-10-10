"""End-to-end pipeline: retrieve -> features -> LightGBM -> decide. Contract: docs/WORKERS.md.

    # validation experiment: train on S1 from folds 1-4 (OOF), tune decisions on that OOF, score the dev subset
    python -m src.er_pipeline validate --exp E003-baseline --k 10 --train-s1 200000
    # full test run: train on all train S1 folds, predict test, write both output TSVs
    python -m src.er_pipeline predict --exp E003-baseline --k 10 --out submissions/sub02_E003

Leakage rules (docs/DECISIONS.md, review): decisions/thresholds are tuned on training-fold OOF only; the dev S1s
never influence any fitted quantity. Retrieval searches the FULL S2/S3 pool of the split, restricted to the same
country string.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.er_blocking import blocking_recall, dense_candidates
from src.er_data import cache_dir, load
from src.er_decide import decide, decide_expected_f, tune_threshold
from src.er_embed import emb_dir
from src.er_features import build_features
from src.er_model import importance, predict, train_oof
from src.er_stage2 import cluster_features
from src.metrics import er_fbeta_macro

VIEWS = ("name", "addr", "both")


def log(msg: str) -> None:
    """Timestamped progress line on stdout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pool_embeddings(split: str, view: str, model: str) -> np.ndarray:
    """S2+S3 embeddings stacked in pool row order, written once to disk and memory-mapped (no RAM copy)."""
    path = emb_dir() / f"{split}_pool_{view}_{model}.npy"
    if not path.exists():
        e2 = np.load(emb_dir() / f"{split}_s2_{view}_{model}.npy", mmap_mode="r")
        e3 = np.load(emb_dir() / f"{split}_s3_{view}_{model}.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(path, mode="w+", dtype=e2.dtype, shape=(len(e2) + len(e3), e2.shape[1]))
        out[:len(e2)] = e2
        out[len(e2):] = e3
        out.flush()
        del out
    return np.load(path, mmap_mode="r")


class Split:
    """All cached data for one split: raw + normalised frames and embedding matrices (memory-mapped)."""

    def __init__(self, split: str, model: str = "small", views=VIEWS):
        """Load raw and normalised frames and memory-map the embedding views of one split."""
        self.split = split
        s1, s2, s3, gt = load(split, with_gt=(split == "train"))
        self.s1, self.pool = s1, pd.concat([s2, s3], ignore_index=True)
        self.gt = gt
        n1 = pd.read_parquet(cache_dir() / f"{split}_s1_norm.parquet")
        n2 = pd.read_parquet(cache_dir() / f"{split}_s2_norm.parquet")
        n3 = pd.read_parquet(cache_dir() / f"{split}_s3_norm.parquet")
        self.left = n1.set_index("entity_id")
        self.right = pd.concat([n2, n3], ignore_index=True).set_index("entity_id")
        # a view spec "<view>_<tag>" (e.g. "both_ft") reads the files embedded with tag <tag> instead of `model`
        spec = {v: (v.partition("_")[0], v.partition("_")[2] or model) for v in views}
        self.emb1 = {v: np.load(emb_dir() / f"{split}_s1_{b}_{t}.npy", mmap_mode="r") for v, (b, t) in spec.items()}
        self.embp = {v: pool_embeddings(split, b, t) for v, (b, t) in spec.items()}
        self.views = tuple(views)
        self.n2 = len(s2)

    def truth(self, s1_ids) -> dict[str, set[str]]:
        """Ground-truth match sets for the given S1 ids (empty set for singletons)."""
        g = self.gt[self.gt.s1_id.isin(set(s1_ids))]
        y = {s: set() for s in s1_ids}
        for s, c in zip(g.s1_id.to_numpy(), g.cand_id.to_numpy()):
            y[s].add(c)
        return y


def retrieve(d: Split, s1_mask: np.ndarray, k: int) -> pd.DataFrame:
    """Union of per-view, per-source top-k candidates; one row per unique pair with rank_<view> columns."""
    q = d.s1[s1_mask].reset_index(drop=True)
    parts = []
    for view in d.views:
        e1 = np.ascontiguousarray(d.emb1[view][s1_mask])
        for sl in (slice(0, d.n2), slice(d.n2, None)):  # S2 and S3 searched separately
            pool = d.pool.iloc[sl].reset_index(drop=True)
            c = dense_candidates(q, pool, e1, np.ascontiguousarray(d.embp[view][sl]), k=k, view=view)
            parts.append(c)
    c = pd.concat(parts, ignore_index=True)
    wide = c.pivot_table(index=["s1_id", "cand_id"], columns="view", values="rank", aggfunc="min")
    wide.columns = [f"rank_{v}" for v in wide.columns]
    return wide.reset_index().fillna(k)   # "not retrieved by this view" = rank k


def featurize(d: Split, pairs: pd.DataFrame) -> pd.DataFrame:
    """Pair features for retrieved pairs of a split (see src/er_features.py)."""
    left_pos = d.left.index.get_indexer(pairs.s1_id)
    right_pos = d.right.index.get_indexer(pairs.cand_id)
    emb = {v: (d.emb1[v], d.embp[v]) for v in d.views}
    # build_features indexes emb by left/right row order, which matches the cached parquet order
    assert (left_pos >= 0).all() and (right_pos >= 0).all()
    return build_features(pairs, d.left, d.right, emb)


def topk_mask(p: pd.DataFrame, k: int) -> np.ndarray:
    """Boolean mask of the top-k rows per S1 by probability (ties broken by order)."""
    return (p.groupby("s1_id", sort=False).prob.rank(ascending=False, method="first") <= k).to_numpy()


def evaluate_strategies(p_tr: pd.DataFrame, y_tr: dict, p_dev: pd.DataFrame, y_dev: dict) -> dict:
    """Tune each decision strategy on TRAINING OOF, then score it once on dev."""
    res = {}
    t, f, _ = tune_threshold(p_tr, y_tr, assign=True)
    res["threshold+assign"] = {"t": t, "train_oof": f, "dev": er_fbeta_macro(y_dev, decide(p_dev, t, True))}
    t2, f2, _ = tune_threshold(p_tr, y_tr, assign=False)
    res["threshold_no_assign"] = {"t": t2, "train_oof": f2, "dev": er_fbeta_macro(y_dev, decide(p_dev, t2, False))}
    best = (-1, None, None)
    for fb in (0.1, 0.2, 0.3):
        tt, ff, _ = tune_threshold(p_tr, y_tr, assign=True, top1_fallback=fb)
        if ff > best[0]:
            best = (ff, tt, fb)
    ff, tt, fb = best
    res["threshold+assign+top1"] = {"t": tt, "fallback": fb, "train_oof": ff,
                                    "dev": er_fbeta_macro(y_dev, decide(p_dev, tt, True, fb))}
    sub_tr = p_tr[p_tr.s1_id.isin(list(y_tr)[:50_000])]
    y_sub = {k: y_tr[k] for k in list(y_tr)[:50_000]}
    res["expected_f+assign"] = {"train_oof": er_fbeta_macro(y_sub, decide_expected_f(sub_tr)),
                                "dev": er_fbeta_macro(y_dev, decide_expected_f(p_dev))}
    return res


def cmd_validate(a: argparse.Namespace) -> None:
    """Validation experiment: fit on folds 1-4 sample, tune decisions on OOF, score the dev subset."""
    import os

    from monitor import Heartbeat

    out = Path("runs") / a.exp
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hb = Heartbeat(os.environ.get("RUN_ID", a.exp), total_steps=6, every_steps=1, metric_name="dev_f05")
    hb.step(0, force=True, stage="load")
    d = Split("train", a.model)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").reindex(d.s1.entity_id)
    rng = np.random.default_rng(42)
    tr_ids = rng.choice(np.flatnonzero(folds.fold.to_numpy() != 0), size=a.train_s1, replace=False)
    tr_mask = np.zeros(len(d.s1), bool)
    tr_mask[tr_ids] = True
    dev_mask = folds.dev.to_numpy().astype(bool)
    log(f"train S1 {tr_mask.sum():,} (folds 1-4), dev S1 {dev_mask.sum():,} (fold 0)")

    res: dict = {"exp": a.exp, "k": a.k, "model": a.model, "train_s1": int(tr_mask.sum()), "dev_s1": int(dev_mask.sum())}
    frames = {}
    for name, mask in (("train", tr_mask), ("dev", dev_mask)):
        t = time.time()
        pairs = retrieve(d, mask, a.k)
        ids = d.s1.entity_id[mask]
        rec = blocking_recall(pairs.assign(rank=pairs.filter(like="rank_").min(1)), d.gt, ids, ks=(a.k,))
        log(f"{name}: {len(pairs):,} pairs ({len(pairs) / mask.sum():.1f}/S1), union recall {rec[f'recall@{a.k}']:.4f} "
            f"[{time.time() - t:.0f}s]")
        res[f"{name}_pairs_per_s1"] = len(pairs) / mask.sum()
        res[f"{name}_blocking_recall"] = rec[f"recall@{a.k}"]
        t = time.time()
        X = featurize(d, pairs)
        truth = set(zip(d.gt.s1_id, d.gt.cand_id))
        X["y"] = [(s, c) in truth for s, c in zip(X.s1_id, X.cand_id)]
        log(f"{name}: features {X.shape} [{time.time() - t:.0f}s]")
        frames[name] = (X, ids)
        hb.step(1 if name == "train" else 2, force=True, stage=f"{name} features done")

    Xtr, tr_ids_s = frames["train"]
    Xdev, dev_ids_s = frames["dev"]
    groups = folds.fold.reindex(Xtr.s1_id).to_numpy()
    t = time.time()
    oof, models = train_oof(Xtr, Xtr.y.to_numpy().astype(int), groups,
                            {"learning_rate": a.lr}, num_boost_round=a.rounds, early_stopping=50)
    log(f"LightGBM: {len(models)} fold models [{time.time() - t:.0f}s]")
    hb.step(3, force=True, stage="stage 1 trained")
    p_tr = Xtr[["s1_id", "cand_id"]].assign(prob=oof)
    p_dev = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models, Xdev))
    y_tr, y_dev = d.truth(tr_ids_s), d.truth(dev_ids_s)
    if a.stage2:
        # stage-1 dev score for the record, then a second LightGBM on stage-1 features + cluster features
        t1, _, _ = tune_threshold(p_tr, y_tr, assign=True)
        res["stage1_dev_f05"] = er_fbeta_macro(y_dev, decide(p_dev, t1, True))
        log(f"stage 1: dev F0.5 {res['stage1_dev_f05']:.4f} (t={t1})")
        t = time.time()
        if a.prefilter:
            # keep top-k candidates per S1 by stage-1 prob (dev: top-15 keeps 99.99% of retrieved true pairs)
            m_tr, m_dev = topk_mask(p_tr, a.prefilter), topk_mask(p_dev, a.prefilter)
            Xtr, p_tr, groups = Xtr[m_tr].reset_index(drop=True), p_tr[m_tr].reset_index(drop=True), groups[m_tr]
            Xdev, p_dev = Xdev[m_dev].reset_index(drop=True), p_dev[m_dev].reset_index(drop=True)
            res["prefilter"] = a.prefilter
            log(f"prefilter top-{a.prefilter}: train {len(Xtr):,} pairs, dev {len(Xdev):,} pairs")
        emb_r = {v: d.embp[v] for v in d.views}
        F_tr = cluster_features(p_tr, d.right, emb_r)
        F_dev = cluster_features(p_dev, d.right, emb_r)
        X2_tr = pd.concat([Xtr.reset_index(drop=True), F_tr], axis=1)
        X2_dev = pd.concat([Xdev.reset_index(drop=True), F_dev], axis=1)
        log(f"cluster features {F_tr.shape[1]} [{time.time() - t:.0f}s]")
        oof, models = train_oof(X2_tr, X2_tr.y.to_numpy().astype(int), groups,
                                {"learning_rate": a.lr}, num_boost_round=a.rounds, early_stopping=50)
        p_tr = X2_tr[["s1_id", "cand_id"]].assign(prob=oof)
        p_dev = X2_dev[["s1_id", "cand_id"]].assign(prob=predict(models, X2_dev))
        log(f"stage 2 trained [{time.time() - t:.0f}s]")
    res["strategies"] = evaluate_strategies(p_tr, y_tr, p_dev, y_dev)
    best = max(res["strategies"].items(), key=lambda kv: kv[1]["train_oof"])
    res["chosen"] = best[0]
    res["dev_f05"] = best[1]["dev"]
    # per-country dev score for the chosen strategy (diagnostic only)
    ctry = d.s1.set_index("entity_id").country
    chosen_pred = (decide_expected_f(p_dev) if best[0].startswith("expected") else
                   decide(p_dev, best[1]["t"], "no_assign" not in best[0], best[1].get("fallback")))
    res["dev_f05_by_country"] = {c: er_fbeta_macro({k: v for k, v in y_dev.items() if ctry[k] == c}, chosen_pred)
                                 for c in ctry.reindex(list(y_dev)).unique()}
    res["top_features"] = importance(models).head(15).round(4).to_dict()
    res["runtime_s"] = round(time.time() - t0)
    p_dev.to_parquet(out / "dev_probs.parquet", index=False)
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    hb.val(res["dev_f05"])
    hb.finish("completed")
    log(json.dumps({k: res[k] for k in ("chosen", "dev_f05", "dev_f05_by_country", "dev_blocking_recall")}, indent=1))


def choose_decision(p: pd.DataFrame, y: dict) -> tuple[str, float, float]:
    """Pick the decision rule on OUT-OF-FOLD predictions only: global threshold vs expected-F0.5 (both with the
    one-S1-per-record assignment). Returns (rule, threshold, oof_score on the compared subset)."""
    t, f_thr, _ = tune_threshold(p, y, assign=True)
    ids = list(y)[:50_000]
    sub = p[p.s1_id.isin(set(ids))]
    y_sub = {k: y[k] for k in ids}
    f_ef = er_fbeta_macro(y_sub, decide_expected_f(sub))
    f_thr_sub = er_fbeta_macro(y_sub, decide(sub, t, True))
    return ("expected_f", t, f_ef) if f_ef > f_thr_sub else ("threshold", t, f_thr)


def apply_decision(p: pd.DataFrame, rule: str, t: float) -> dict[str, set[str]]:
    """Turn pair probabilities into match sets with the chosen rule (threshold or expected F0.5)."""
    return decide_expected_f(p) if rule == "expected_f" else decide(p, t, assign=True)


def cmd_predict(a: argparse.Namespace) -> None:
    """Fit stage 1 (+ optional stage 2) on train S1 with grouped OOF, choose the decision rule on OOF, predict test.

    Stage 2 on test: chunks partition the S1s, so sibling features are computed per chunk; the record-level
    competition features (best other S1) need all chunks and are recomputed once over every test pair.
    """
    import os

    from monitor import Heartbeat
    from src.er_data import dataset_dir
    from src.er_stage2 import competition_features
    from src.er_submission import write_outputs

    t0 = time.time()
    out_dir = Path(a.out)
    run_dir = Path("runs") / a.exp
    (run_dir / "test_chunks").mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(os.environ.get("RUN_ID", a.exp), total_steps=24, every_steps=1, metric_name="train_oof_f05")
    hb.step(0, force=True, stage="load train")
    d = Split("train", a.model)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").reindex(d.s1.entity_id)
    rng = np.random.default_rng(42)
    tr_mask = np.zeros(len(d.s1), bool)
    tr_mask[rng.choice(len(d.s1), size=a.train_s1, replace=False)] = True
    pairs = retrieve(d, tr_mask, a.k)
    X = featurize(d, pairs)
    truth = set(zip(d.gt.s1_id, d.gt.cand_id))
    X["y"] = [(s, c) in truth for s, c in zip(X.s1_id, X.cand_id)]
    log(f"train: {len(X):,} pairs from {tr_mask.sum():,} S1 [{time.time() - t0:.0f}s]")
    hb.step(1, force=True, stage="stage-1 lightgbm")
    groups = folds.fold.reindex(X.s1_id).to_numpy()
    params = {"learning_rate": a.lr}
    oof, models1 = train_oof(X, X.y.to_numpy().astype(int), groups, params, num_boost_round=a.rounds,
                             early_stopping=50)
    y_tr = d.truth(d.s1.entity_id[tr_mask])
    p_tr = X[["s1_id", "cand_id"]].assign(prob=oof)
    models2 = None
    if a.stage2:
        m = topk_mask(p_tr, a.prefilter) if a.prefilter else np.ones(len(p_tr), bool)
        X, p_tr, groups = X[m].reset_index(drop=True), p_tr[m].reset_index(drop=True), groups[m]
        F = cluster_features(p_tr, d.right, {v: d.embp[v] for v in d.views})
        X2 = pd.concat([X, F], axis=1)
        hb.step(2, force=True, stage="stage-2 lightgbm")
        oof2, models2 = train_oof(X2, X2.y.to_numpy().astype(int), groups, params, num_boost_round=a.rounds,
                                  early_stopping=50)
        p_tr = X2[["s1_id", "cand_id"]].assign(prob=oof2)
        del X2, F
    rule, t, f = choose_decision(p_tr, y_tr)
    log(f"stage{'2' if a.stage2 else '1'} OOF F0.5 {f:.4f} with rule={rule} (t={t}) [{time.time() - t0:.0f}s]")
    hb.val(f)
    for name, ms in (("s1", models1), ("s2", models2 or [])):
        for i, mdl in enumerate(ms):
            mdl.save_model(str(run_dir / f"lgbm_{name}_fold{i}.txt"))
    del d, X, pairs

    te = Split("test", a.model)
    s1_ids = te.s1.entity_id.to_numpy()
    n_chunks = (len(te.s1) + a.chunk - 1) // a.chunk
    probs_parts = []
    for ci, start in enumerate(range(0, len(te.s1), a.chunk)):
        mask = np.zeros(len(te.s1), bool)
        mask[start:start + a.chunk] = True
        Xt = featurize(te, retrieve(te, mask, a.k))
        pt = Xt[["s1_id", "cand_id"]].assign(prob=predict(models1, Xt).astype(np.float32))
        if a.stage2:
            m = topk_mask(pt, a.prefilter) if a.prefilter else np.ones(len(pt), bool)
            Xt, pt = Xt[m].reset_index(drop=True), pt[m].reset_index(drop=True)
            F = cluster_features(pt, te.right, {v: te.embp[v] for v in te.views})   # sibling feats are chunk-local
            pd.concat([Xt, F], axis=1).to_parquet(run_dir / "test_chunks" / f"{ci:03d}.parquet", index=False)
        probs_parts.append(pt)
        log(f"test chunk {ci + 1}/{n_chunks}: {len(pt):,} pairs kept [{time.time() - t0:.0f}s]")
        hb.step(3 + ci, force=True, stage="test stage 1")
    probs = pd.concat(probs_parts, ignore_index=True)
    if a.stage2:
        # record-level competition features need ALL test pairs: recompute globally, then stage-2 predict per chunk
        comp = competition_features(probs)
        off, parts2 = 0, []
        for ci in range(n_chunks):
            X2 = pd.read_parquet(run_dir / "test_chunks" / f"{ci:03d}.parquet")
            n = len(X2)
            for c, v in comp.items():
                X2[c] = v[off:off + n]
            parts2.append(X2[["s1_id", "cand_id"]].assign(prob=predict(models2, X2).astype(np.float32)))
            off += n
            hb.step(3 + n_chunks + ci, force=True, stage="test stage 2")
        probs = pd.concat(parts2, ignore_index=True)
        log(f"stage 2 applied to {len(probs):,} test pairs [{time.time() - t0:.0f}s]")
    probs.to_parquet(run_dir / "test_probs.parquet", index=False)
    matches = apply_decision(probs, rule, t)
    candidates: dict[str, list[str]] = {}
    for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
        candidates.setdefault(s, []).append(c)
    write_outputs(matches, candidates, s1_ids, out_dir, test_dir=dataset_dir() / "test")
    n_nonempty = sum(1 for v in matches.values() if v)
    info = {"exp": a.exp, "stage2": bool(a.stage2), "prefilter": a.prefilter, "rule": rule, "threshold": t,
            "train_oof_f05": f, "train_s1": int(a.train_s1), "k": a.k, "test_s1": len(s1_ids),
            "test_pairs": len(probs), "nonempty_share": n_nonempty / len(s1_ids),
            "mean_matches": float(np.mean([len(matches.get(x, ())) for x in s1_ids])),
            "runtime_s": round(time.time() - t0)}
    (run_dir / "predict.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(json.dumps(info))
    hb.finish("completed")


def main() -> None:
    """CLI entry point: validate / predict subcommands."""
    ap = argparse.ArgumentParser(prog="python -m src.er_pipeline")
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate")
    v.add_argument("--exp", required=True)
    v.add_argument("--k", type=int, default=10, help="top-k per view per source")
    v.add_argument("--train-s1", type=int, default=200_000)
    v.add_argument("--lr", type=float, default=0.1)
    v.add_argument("--rounds", type=int, default=1000)
    v.add_argument("--stage2", action="store_true", help="add second-stage cluster features (E005)")
    v.add_argument("--prefilter", type=int, default=0, help="stage 2 on top-k stage-1 candidates per S1 (0 = all)")
    v.add_argument("--model", default="small")
    pr = sp.add_parser("predict")
    pr.add_argument("--exp", required=True)
    pr.add_argument("--out", required=True, help="folder for matching_results.tsv + candidate_pairs.tsv")
    pr.add_argument("--k", type=int, default=10)
    pr.add_argument("--train-s1", type=int, default=300_000)
    pr.add_argument("--chunk", type=int, default=200_000, help="test S1 per chunk (memory)")
    pr.add_argument("--lr", type=float, default=0.1)
    pr.add_argument("--rounds", type=int, default=1000)
    pr.add_argument("--model", default="small")
    pr.add_argument("--stage2", action="store_true")
    pr.add_argument("--prefilter", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "validate":
        cmd_validate(a)
    elif a.cmd == "predict":
        cmd_predict(a)


if __name__ == "__main__":
    main()
