"""Full-population stage 1 + stage 2 (fixes the cross-S1 feature mismatch, docs/DECISIONS.md 2026-09-25).

Cross-S1 features (a record's competing S1s) depend on how many S1s are scored, so every split must be scored in
full: all 2.2M train S1 (out-of-fold) and all 1.7M test S1. Then stage 2 is trained/validated/applied in the same
regime as test.

    python -m src.er_fullpass stage1 --exp E007 --train-s1 200000          # ~2 h: fit, score all train + test
    python -m src.er_fullpass stage2 --exp E007 --out submissions/sub03    # validate on dev, then predict test

Stage-1 fitting uses S1s from folds 1-4 only, so dev (fold 0) stays untouched: a train S1 in fold g is scored by
the model that excluded fold g; fold-0 and test S1s are scored by the mean of all fold models.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.er_data import cache_dir
from src.er_model import predict, train_oof
from src.er_decide import decide
from src.er_pipeline import VIEWS, Split, apply_decision, choose_decision, featurize, log, retrieve, topk_mask
from src.er_stage2 import cluster_features, competition_features
from src.metrics import er_fbeta_macro

COMP = ("s2_other_s1_best", "s2_margin_vs_other_s1")


def run_dir(exp: str) -> Path:
    """Run folder runs/<exp> (created if missing)."""
    d = Path("runs") / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_stage1(a: argparse.Namespace) -> None:
    """Fit stage 1 on the chosen pool, then score every train (out-of-fold) and test S1 in resumable chunks."""
    from monitor import Heartbeat

    t0 = time.time()
    rd = run_dir(a.exp)
    hb = Heartbeat(os.environ.get("RUN_ID", a.exp), total_steps=25, every_steps=1)
    d = Split("train", a.model, a.views)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").reindex(d.s1.entity_id)
    fold_arr = folds.fold.to_numpy()
    rng = np.random.default_rng(42)
    fit_mask = np.zeros(len(d.s1), bool)
    if a.fit_pool == "fold0":
        # fold 0 minus dev: S1s no learned retrieval view (E014 bi-encoder, folds 1-4) or CE (folds 1-2) has seen,
        # so their embedding/CE features look like test. OOF uses 4 random internal groups (by S1).
        cand = np.flatnonzero((fold_arr == 0) & ~folds.dev.to_numpy())
        fit_mask[rng.choice(cand, size=min(a.train_s1, len(cand)), replace=False)] = True
        group_arr = np.full(len(d.s1), -1)
        group_arr[fit_mask] = np.random.default_rng(0).integers(0, 4, fit_mask.sum())
    else:
        fit_mask[rng.choice(np.flatnonzero(fold_arr != 0), size=a.train_s1, replace=False)] = True
        group_arr = fold_arr
    truth = set(zip(d.gt.s1_id, d.gt.cand_id))
    saved = sorted(rd.glob("stage1_fold*.txt"))
    if saved:
        # resume after a stop: reuse the fitted fold models (saved at best iteration), keep already-scored chunks
        import lightgbm as lgb
        fold_ids = [int(p.stem.removeprefix("stage1_fold")) for p in saved]
        models = [lgb.Booster(model_file=str(p)) for p in saved]
        log(f"reusing {len(models)} saved stage-1 models, folds {fold_ids}")
    else:
        X = featurize(d, retrieve(d, fit_mask, a.k))
        X["y"] = [(s, c) in truth for s, c in zip(X.s1_id, X.cand_id)]
        groups = pd.Series(group_arr, index=d.s1.entity_id).reindex(X.s1_id).to_numpy()
        _, models = train_oof(X, X.y.to_numpy().astype(int), groups, {"learning_rate": a.lr},
                              num_boost_round=a.rounds, early_stopping=50)
        fold_ids = [int(g) for g in np.unique(groups)]  # models[i] excluded fold fold_ids[i]
        for i, m in enumerate(models):
            m.save_model(str(rd / f"stage1_fold{fold_ids[i]}.txt"))
        log(f"stage 1 fitted on {fit_mask.sum():,} S1 ({len(X):,} pairs), folds {fold_ids} [{time.time() - t0:.0f}s]")
        del X
    hb.step(1, force=True, stage="score train")

    def score(split_obj: Split, name: str, fold_of_s1: np.ndarray | None, labels: set | None, step0: int) -> None:
        """Retrieve, featurise and score one split chunk by chunk, keeping the top candidates per S1."""
        out = rd / f"{name}_chunks"
        out.mkdir(exist_ok=True)
        n = min(len(split_obj.s1), a.max_s1) if a.max_s1 else len(split_obj.s1)   # --max-s1: smoke tests only
        for ci, start in enumerate(range(0, n, a.chunk)):
            path = out / f"{ci:03d}.parquet"
            if path.exists():
                continue                                      # resumable
            mask = np.zeros(len(split_obj.s1), bool)
            mask[start:min(start + a.chunk, n)] = True
            Xc = featurize(split_obj, retrieve(split_obj, mask, a.k))
            prob = predict(models, Xc)                        # default: mean of fold models
            if fold_of_s1 is not None:                        # train: out-of-fold model per S1's fold
                f = pd.Series(fold_of_s1, index=split_obj.s1.entity_id).reindex(Xc.s1_id).to_numpy()
                cols = models[0].feature_name()
                for i, fid in enumerate(fold_ids):
                    sel = f == fid
                    if sel.any():
                        prob[sel] = models[i].predict(Xc.loc[sel, cols], num_iteration=models[i].best_iteration)
            Xc["prob"] = prob.astype(np.float32)
            Xc = Xc[topk_mask(Xc, a.prefilter)].reset_index(drop=True)
            if labels is not None:
                Xc["y"] = [(s, c) in labels for s, c in zip(Xc.s1_id, Xc.cand_id)]
            Xc.to_parquet(path, index=False)
            log(f"{name} chunk {ci + 1}: kept {len(Xc):,} pairs [{time.time() - t0:.0f}s]")
            hb.step(step0 + ci, force=True, stage=f"score {name}")

    score(d, "train", group_arr, truth, 2)          # group -1 (not fitted on) -> mean of all models
    del d
    te = Split("test", a.model, a.views)
    score(te, "test", None, None, 14)
    (rd / "stage1.json").write_text(json.dumps({"fold_ids": [int(x) for x in fold_ids], "k": a.k, "views": a.views,
                                                "prefilter": a.prefilter, "train_s1": a.train_s1,
                                                "runtime_s": round(time.time() - t0)}), encoding="utf-8")
    hb.finish("completed")
    log(f"done [{time.time() - t0:.0f}s]")


def load_chunks(rd: Path, name: str, columns=None) -> list[pd.DataFrame]:
    """Load a split's stage-1 chunk parquets in order."""
    return [pd.read_parquet(p, columns=columns) for p in sorted((rd / f"{name}_chunks").glob("*.parquet"))]


def prune_rows(X: pd.DataFrame, split: str, eps: float, ce_dir: str = "", ce_eps: float = 0.0,
               unseen_ce_dir: str = "", unseen_ce: float = 0.0) -> pd.DataFrame:
    """The final model's candidate set (= candidate_pairs.tsv): keep a pair when stage 1 finds it plausible (prob >= eps)
    OR, with ce_dir, the cross-encoder does (score >= ce_eps). Features are built on the full stage-1 top-15 first, so
    every split sees identically computed features; only the rows the final model trains/predicts on are filtered.

    Organisers (26 Sep): a smaller candidate set per S1 ranks higher. Dev (E017): stage-1 >= 0.003 alone gives 6.43
    candidates/S1 (recall 0.9961, F0.5 -0.00007); stage-1 >= 0.02 OR CE(E016) >= 0.01 gives 5.28/S1 (recall 0.9966,
    F0.5 unchanged). The CE is a filter stage of the candidate cascade, scored on every stage-2 row (fold 0 + test).
    """
    if eps <= 0:
        return X
    keep = X.prob.to_numpy() >= eps
    if ce_dir:
        ce = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "ce_score"])
                        for f in sorted((Path(ce_dir) / f"{split}_ce").glob("*.parquet"))], ignore_index=True)
        v = X[["s1_id", "cand_id"]].merge(ce, on=["s1_id", "cand_id"], how="left").ce_score.to_numpy()
        if np.isnan(v).mean() > 0.001:
            raise ValueError(f"{ce_dir} has no score for {np.isnan(v).mean():.2%} of {split} stage-2 rows")
        keep |= np.nan_to_num(v, nan=0.0) >= ce_eps
    if unseen_ce_dir and unseen_ce > 0:
        # E037: for S1 countries absent from train, also keep pairs that a cross-encoder adapted to that country scores
        # >= unseen_ce (the filter CE above never saw its text). Countries come from the data; nothing is hard-coded.
        from src.er_data import load
        seen = set(pd.read_parquet(cache_dir() / "train_s1.parquet", columns=["country"]).country.unique())
        ctry = load(split, with_gt=False)[0].set_index("entity_id").country
        unseen = ~ctry.reindex(X.s1_id).isin(seen).to_numpy()
        if unseen.any():
            ce = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "ce_score"])
                            for f in sorted((Path(unseen_ce_dir) / f"{split}_ce").glob("*.parquet"))], ignore_index=True)
            v = X[["s1_id", "cand_id"]].merge(ce, on=["s1_id", "cand_id"], how="left").ce_score.to_numpy()
            add = unseen & ~keep & (np.nan_to_num(v, nan=0.0) >= unseen_ce)
            log(f"{split}: unseen-country rescue adds {int(add.sum()):,} pairs ({add.sum() / max(unseen.sum(), 1):.4f} of their rows)")
            keep |= add
    log(f"{split}: candidate set {keep.sum() / X.s1_id.nunique():.2f}/S1 (was {len(X) / X.s1_id.nunique():.2f})")
    return X[keep].reset_index(drop=True)


def with_ce(chunks: list[pd.DataFrame], ce_dirs, split: str) -> list[pd.DataFrame]:
    """Attach cross-encoder scores (src.er_crossenc score output: one parquet per chunk, same row order).

    Several CE dirs -> columns ce_score, ce_score_2, ... (one feature per cross-encoder)."""
    if isinstance(ce_dirs, str):
        ce_dirs = [ce_dirs] if ce_dirs else []
    for i, ce_dir in enumerate(ce_dirs):
        col = "ce_score" if i == 0 else f"ce_score_{i + 1}"
        files = sorted((Path(ce_dir) / f"{split}_ce").glob("*.parquet"))
        if len(files) != len(chunks):
            raise ValueError(f"{len(files)} CE files for {len(chunks)} {split} chunks in {ce_dir}")
        out = []
        for c, f in zip(chunks, files):
            ce = pd.read_parquet(f)
            if len(ce) != len(c) or not (ce.s1_id.to_numpy() == c.s1_id.to_numpy()).all() \
                    or not (ce.cand_id.to_numpy() == c.cand_id.to_numpy()).all():
                raise ValueError(f"CE rows misaligned with {f.name} ({ce_dir})")
            out.append(c.assign(**{col: ce.ce_score.to_numpy(np.float32)}))
        chunks = out
    return chunks


def name_rarity(split_obj: Split) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free name-rarity signals per record, computed over the whole split (S1 and S2+S3 separately).

    Error analysis (E010 dev): many remaining errors are candidates with an EMPTY address, where only the name can
    decide; a unique name is almost surely a match, a common one is ambiguous. Pool counts are normalised by the
    pool/S1 size ratio per country, because test has ~23% more S2+S3 records per S1 than train.
    Returns (left_feats indexed by S1 id, right_feats indexed by S2/S3 id).
    """
    L, R = split_obj.left, split_obj.right
    lk = L.country.astype(str) + "|" + L.name_core.astype(str)
    rk = R.country.astype(str) + "|" + R.name_core.astype(str)
    s1_cnt, pool_cnt = lk.value_counts(), rk.value_counts()
    ratio = (R.country.value_counts() / L.country.value_counts()).fillna(1.0)
    # rarest core token of each record: document frequency over S1 + pool within the country
    toks = pd.concat([L[["country", "name_core"]], R[["country", "name_core"]]])
    ex = toks.assign(tok=toks.name_core.str.split()).explode("tok").dropna(subset=["tok"])
    df = (ex.country.astype(str) + "|" + ex.tok.astype(str)).value_counts()

    def min_df(frame: pd.DataFrame) -> np.ndarray:
        """Per record: document frequency of its rarest core-name token within its country."""
        e = frame.assign(tok=frame.name_core.str.split()).explode("tok")
        v = (e.country.astype(str) + "|" + e.tok.astype(str)).map(df).fillna(0.0)
        return v.groupby(level=0, sort=False).min().reindex(frame.index).to_numpy(np.float32)

    lr = ratio.reindex(L.country).to_numpy()
    rr = ratio.reindex(R.country).to_numpy()
    left = pd.DataFrame({
        "rar_s1_same_name_s1": lk.map(s1_cnt).to_numpy(np.float32),
        "rar_s1_same_name_pool": (lk.map(pool_cnt).fillna(0).to_numpy() / lr).astype(np.float32),
        "rar_s1_min_tok_df": min_df(L),
    }, index=L.index)
    right = pd.DataFrame({
        "rar_c_same_name_pool": (rk.map(pool_cnt).to_numpy() / rr).astype(np.float32),
        "rar_c_same_name_s1": rk.map(s1_cnt).fillna(0).to_numpy(np.float32),
        "rar_c_min_tok_df": min_df(R),
    }, index=R.index)
    return left, right


def _norm2(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v2 normalised strings (src.er_norm2 cache), indexed by entity_id: (S1, S2+S3)."""
    L = pd.read_parquet(cache_dir() / f"{split}_s1_norm2.parquet").set_index("entity_id")
    R = pd.concat([pd.read_parquet(cache_dir() / f"{split}_s{k}_norm2.parquet") for k in (2, 3)]).set_index("entity_id")
    return L, R


def norm2_pair_features(s1_ids, cand_ids, L2: pd.DataFrame, R2: pd.DataFrame, only: tuple | None = None) -> dict:
    """E024 features from v2 strings: street types expanded, dotted legal forms collapsed (src/er_norm2.py)."""
    from rapidfuzz import fuzz, process
    a_n, b_n = L2.name_core2.reindex(s1_ids).to_numpy(), R2.name_core2.reindex(cand_ids).to_numpy()
    a_a, b_a = L2.addr_norm2.reindex(s1_ids).to_numpy(), R2.addr_norm2.reindex(cand_ids).to_numpy()
    spec = {"name2_ratio": (a_n, b_n, fuzz.ratio), "name2_tset": (a_n, b_n, fuzz.token_set_ratio),
            "name2_tsort": (a_n, b_n, fuzz.token_sort_ratio), "addr2_tset": (a_a, b_a, fuzz.token_set_ratio),
            "addr2_tsort": (a_a, b_a, fuzz.token_sort_ratio)}
    return {k: (process.cpdist(x, y, scorer=f, workers=-1) / 100.0).astype(np.float32)
            for k, (x, y, f) in spec.items() if only is None or k in only}


def density_mask(allp: pd.DataFrame, always: set | None, keep_frac: float, seed: int = 11) -> np.ndarray:
    """Rows of allp whose S1 stays in the competitor population: every S1 in `always` (fit + dev) plus a random
    keep_frac of the others. E025: test has ~18% fewer S1s per candidate record than train (2.67 vs 3.27 for US and
    India), so competition features learned on the full train population mean something else on test; dropping
    never-fitted S1s turns their records into extra distractors, the way test looks."""
    if keep_frac >= 1.0:
        return np.ones(len(allp), bool)
    ids = pd.Index(allp.s1_id.unique())
    keep = pd.Series(np.random.default_rng(seed).random(len(ids)) < keep_frac, index=ids)
    if always:
        keep[keep.index.isin(list(always))] = True
    return keep.reindex(allp.s1_id).to_numpy()


def stage2_frames(split_obj: Split, chunks: list[pd.DataFrame], keep_s1: set | None,
                  comp_cols: tuple[str, ...] = (), norm2: bool = False, comp_keep: float = 1.0) -> pd.DataFrame:
    """Stage-2 design matrix: sibling feats per chunk (chunks partition S1) + competition feats over ALL chunks.

    comp_cols: extra stage-1 columns (e.g. name_full_tset) whose record-level "best other S1" competition features
    are added, computed over the full population like the prob ones."""
    allp = pd.concat([c[["s1_id", "cand_id", "prob", *comp_cols]] for c in chunks], ignore_index=True)
    # train only (keep_s1 given): test keeps its full, real population
    dm = density_mask(allp, keep_s1, comp_keep if keep_s1 is not None else 1.0)
    if not dm.all():
        # competitors come from the reduced population; rows of dropped S1s still get features but are never used
        full = allp
        allp = allp[dm].reset_index(drop=True)
    comp = competition_features(allp)
    for col in comp_cols:
        comp.update(competition_features(allp, col))
    if norm2:
        L2, R2 = _norm2(split_obj.split)
        allp["name2_ratio"] = norm2_pair_features(allp.s1_id, allp.cand_id, L2, R2, only=("name2_ratio",))["name2_ratio"]
        comp.update(competition_features(allp, "name2_ratio"))
    comp_keys = list(comp)
    if not dm.all():                                       # scatter back to the full row order (dropped rows: NaN)
        for k in comp_keys:
            v = np.full(len(full), np.nan, np.float32)
            v[dm] = comp[k]
            comp[k] = v
        allp = full
    del allp
    parts, off = [], 0
    emb = {v: split_obj.embp[v] for v in split_obj.views}
    rl, rr = name_rarity(split_obj)
    for c in chunks:
        n = len(c)
        sel = np.ones(n, bool) if keep_s1 is None else c.s1_id.isin(keep_s1).to_numpy()
        if sel.any():
            cs = c[sel].reset_index(drop=True)
            F = cluster_features(cs[["s1_id", "cand_id", "prob"]], split_obj.right, emb)
            for k in comp_keys:
                F[k] = comp[k][off:off + n][sel]
            if norm2:
                for k, v in norm2_pair_features(cs.s1_id, cs.cand_id, L2, R2).items():
                    F[k] = v
            F = pd.concat([F, rl.reindex(cs.s1_id).reset_index(drop=True),
                           rr.reindex(cs.cand_id).reset_index(drop=True)], axis=1)
            parts.append(pd.concat([cs, F], axis=1))
        off += n
    return pd.concat(parts, ignore_index=True)


def stage3_features(split_obj: Split, P: pd.DataFrame, block: int = 200_000) -> pd.DataFrame:
    """Sibling/anchor features recomputed from STAGE-2 probabilities (better anchors than stage 1), prefixed t3_.

    Per-S1 only: the record-level competition columns are dropped, because stage-2 probabilities exist for a subset
    of train S1s but for every test S1, so a cross-S1 feature would mean different things on train and test
    (docs/DECISIONS.md 2026-09-25). P rows must keep all candidates of each S1 together; computed in S1 blocks.
    """
    P = P[["s1_id", "cand_id", "prob"]].reset_index(drop=True)
    emb = {v: split_obj.embp[v] for v in split_obj.views}
    codes, uniq = pd.factorize(P.s1_id)
    parts = []
    for b0 in range(0, len(uniq), block):
        rows = np.flatnonzero((codes >= b0) & (codes < b0 + block))
        F = cluster_features(P.iloc[rows], split_obj.right, emb).drop(columns=list(COMP))
        F.index = rows
        parts.append(F)
    F = pd.concat(parts).sort_index()
    return F.rename(columns=lambda c: "t3_" + c.removeprefix("s2_")).reset_index(drop=True)


def lgb_params(a: argparse.Namespace) -> dict:
    """LightGBM overrides for stage 2: --lr plus any JSON given with --lgb-params (e.g. '{"num_leaves": 255}')."""
    return {"learning_rate": a.lr, **json.loads(a.lgb_params or "{}")}


def cached_frame(a: argparse.Namespace, name: str, build) -> pd.DataFrame:
    """Stage-2 design matrix from --frames DIR if cached there, else build() and (with --frames) save it.

    The frames depend on the stage-1 chunks, the CE dirs and --comp-cols (and, for train, the fit/dev S1 sample);
    only reuse a cache built with the same ones. frames.json records them and a mismatch raises."""
    if not a.frames:
        return build()
    fd = Path(a.frames)
    fd.mkdir(parents=True, exist_ok=True)
    key = {"exp": a.exp, "ce_dir": list(a.ce_dir), "comp_cols": list(a.comp_cols), "views": list(a.views),
           "fit_folds": list(a.fit_folds), "train_s1": a.train_s1, **({"norm2": True} if getattr(a, "norm2", False) else {}),
           **({"comp_keep": a.comp_keep} if getattr(a, "comp_keep", 1.0) < 1.0 else {}),
}
    meta, path = fd / "frames.json", fd / f"{name}_frame.parquet"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) != key:
        raise ValueError(f"{fd} was built for {meta.read_text()}, not {key}; use another --frames dir")
    if path.exists():
        log(f"loading cached {name} frame from {path}")
        return pd.read_parquet(path)
    X = build()
    X.to_parquet(path, index=False)
    meta.write_text(json.dumps(key), encoding="utf-8")
    log(f"saved {name} frame {X.shape} -> {path}")
    return X


def cmd_stage2(a: argparse.Namespace) -> None:
    """Build (or load cached) stage-2 frames, fit on the clean fit pool, report dev F0.5, optionally predict test."""
    from src.er_data import dataset_dir
    from src.er_submission import write_outputs

    t0 = time.time()
    rd = run_dir(a.exp)
    d = Split("train", a.model, a.views)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id")
    rng = np.random.default_rng(7)
    pool = folds.index[folds.fold.isin(a.fit_folds) & ~folds.dev].to_numpy()   # dev never fits (fold 0 holds dev)
    fit_ids = set(rng.choice(pool, size=min(a.train_s1, len(pool)), replace=False))
    dev_ids = set(folds.index[folds.dev])
    X = cached_frame(a, "train", lambda: stage2_frames(d, with_ce(load_chunks(rd, "train"), a.ce_dir, "train"),
                                                       fit_ids | dev_ids, tuple(a.comp_cols), a.norm2, a.comp_keep))
    log(f"stage-2 frame {X.shape} [{time.time() - t0:.0f}s]")
    X = prune_rows(X, "train", a.prune_eps, a.prune_ce_dir, a.prune_ce, a.prune_unseen_ce_dir, a.prune_unseen_ce)
    is_dev = X.s1_id.isin(dev_ids).to_numpy()
    Xtr, Xdev = X[~is_dev].reset_index(drop=True), X[is_dev].reset_index(drop=True)
    y_tr, y_dev = d.truth(sorted(fit_ids)), d.truth(sorted(dev_ids))
    # stage-1-only reference on the same (full-population, top-k) candidates
    p1_tr = Xtr[["s1_id", "cand_id", "prob"]]
    rule1, t1, _ = choose_decision(p1_tr, y_tr)
    res = {"exp": a.exp, "stage1_dev_f05": er_fbeta_macro(y_dev, apply_decision(Xdev[["s1_id", "cand_id", "prob"]], rule1, t1))}
    feat = Xtr.drop(columns=["prob"])
    feat["s1_prob"] = Xtr.prob.to_numpy()
    groups = folds.fold.reindex(Xtr.s1_id).to_numpy()
    if len(np.unique(groups)) < 2:
        # single fit fold (e.g. --fit-folds 3 so CE scores are out-of-sample): OOF needs >= 2 groups,
        # so split its S1s into 4 random internal groups (still grouped by S1, no pair leakage)
        ids = pd.Index(Xtr.s1_id.unique())
        g = pd.Series(np.random.default_rng(0).integers(0, 4, len(ids)), index=ids)
        groups = g.reindex(Xtr.s1_id).to_numpy()
    oof, models = train_oof(feat, Xtr.y.to_numpy().astype(int), groups, lgb_params(a),
                            num_boost_round=a.rounds, early_stopping=50)
    fdev = Xdev.drop(columns=["prob"])
    fdev["s1_prob"] = Xdev.prob.to_numpy()
    p_tr = Xtr[["s1_id", "cand_id"]].assign(prob=oof)
    p_dev = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models, fdev))
    # keep dev predictions + key features for error analysis (labels: y)
    keep_cols = [c for c in ("y", "prob", "ce_score", "ce_score_2", "name_tset", "addr_tset", "cos_both", "num_first_eq", "post_eq",
                             "s2_margin_vs_other_s1") if c in Xdev.columns]
    Xdev[["s1_id", "cand_id"] + keep_cols].rename(columns={"prob": "s1_prob"}).assign(prob=p_dev.prob.to_numpy()) \
        .to_parquet(rd / f"dev_stage2_{a.tag or 'last'}.parquet", index=False)
    rule, t, f = choose_decision(p_tr, y_tr)
    res.update({"stage2_rule": rule, "stage2_t": t, "stage2_oof": f,
                "dev_f05": er_fbeta_macro(y_dev, apply_decision(p_dev, rule, t))})
    ctry = d.s1.set_index("entity_id").country
    pred = apply_decision(p_dev, rule, t)
    res["dev_by_country"] = {c: er_fbeta_macro({k: v for k, v in y_dev.items() if ctry[k] == c}, pred)
                             for c in ("US", "India")}
    models3 = None
    if a.stage3:
        # stage 3: same frame + anchors re-picked with stage-2 probs (train: OOF, same S1 groups -> no leakage)
        f3 = pd.concat([feat.reset_index(drop=True), stage3_features(d, p_tr)], axis=1)
        f3dev = pd.concat([fdev.reset_index(drop=True), stage3_features(d, p_dev)], axis=1)
        oof3, models3 = train_oof(f3, Xtr.y.to_numpy().astype(int), groups, lgb_params(a),
                                  num_boost_round=a.rounds, early_stopping=50)
        p_tr3 = Xtr[["s1_id", "cand_id"]].assign(prob=oof3)
        p_dev3 = Xdev[["s1_id", "cand_id"]].assign(prob=predict(models3, f3dev))
        rule3, t3, f3oof = choose_decision(p_tr3, y_tr)
        pred3 = apply_decision(p_dev3, rule3, t3)
        res.update({"stage3_rule": rule3, "stage3_t": t3, "stage3_oof": f3oof,
                    "stage3_dev_f05": er_fbeta_macro(y_dev, pred3),
                    "stage3_dev_by_country": {c: er_fbeta_macro({k: v for k, v in y_dev.items() if ctry[k] == c}, pred3)
                                              for c in ("US", "India")}})
        p_dev3.to_parquet(rd / f"dev_stage3_{a.tag or 'last'}.parquet", index=False)
        rule, t = rule3, t3                             # --stage3: the test run uses stage 3
    log(json.dumps(res))
    (rd / "stage2.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    del d, X, Xtr, Xdev
    if not a.out:
        return
    te = Split("test", a.model, a.views)
    Xt = cached_frame(a, "test", lambda: stage2_frames(te, with_ce(load_chunks(rd, "test"), a.ce_dir, "test"), None,
                                                       tuple(a.comp_cols), a.norm2))
    Xt = prune_rows(Xt, "test", a.prune_eps, a.prune_ce_dir, a.prune_ce, a.prune_unseen_ce_dir, a.prune_unseen_ce)
    ft = Xt.drop(columns=["prob"])
    ft["s1_prob"] = Xt.prob.to_numpy()
    probs = Xt[["s1_id", "cand_id"]].assign(prob=predict(models, ft).astype(np.float32))
    probs.to_parquet(rd / "test_probs_stage2.parquet", index=False)
    if models3 is not None:
        ft3 = pd.concat([ft.reset_index(drop=True), stage3_features(te, probs)], axis=1)
        probs = Xt[["s1_id", "cand_id"]].assign(prob=predict(models3, ft3).astype(np.float32))
        probs.to_parquet(rd / "test_probs_stage3.parquet", index=False)
        del ft3
    matches = apply_decision(probs, rule, t)
    cands: dict[str, list[str]] = {}
    for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
        cands.setdefault(s, []).append(c)
    write_outputs(matches, cands, te.s1.entity_id.to_numpy(), Path(a.out), test_dir=dataset_dir() / "test")
    if a.unseen_threshold:
        # Same run, same probabilities and candidates: S1s whose country string never occurs in train get a stricter
        # global threshold (domain shift -> precision first under F0.5). Countries are read from the data, not listed.
        train_countries = set(pd.read_parquet(cache_dir() / "train_s1.parquet", columns=["country"]).country.unique())
        ctry = te.s1.set_index("entity_id").country
        unseen = ~ctry.reindex(probs.s1_id).isin(train_countries).to_numpy()
        unseen_s1 = set(probs.s1_id[unseen])            # built once (inside the comprehension it was O(n^2): hung)
        m2 = {k: v for k, v in matches.items() if k not in unseen_s1}
        m2.update(decide(probs[unseen], a.unseen_threshold, assign=True))
        log(f"unseen-country variant: {int(unseen.sum()):,} pairs of countries {sorted(set(ctry[~ctry.isin(train_countries)]))}")
        write_outputs(m2, cands, te.s1.entity_id.to_numpy(), Path(a.out + "_unseen"), test_dir=dataset_dir() / "test")
    s1 = te.s1.entity_id.to_numpy()
    info = {**res, "test_pairs": len(probs), "nonempty_share": sum(1 for v in matches.values() if v) / len(s1),
            "mean_matches": float(np.mean([len(matches.get(x, ())) for x in s1])), "runtime_s": round(time.time() - t0)}
    (rd / "predict.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(json.dumps(info))


def main() -> None:
    """CLI entry point: stage1 / stage2 subcommands."""
    ap = argparse.ArgumentParser(prog="python -m src.er_fullpass")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stage1", "stage2"):
        p = sp.add_parser(name)
        p.add_argument("--exp", required=True)
        p.add_argument("--model", default="small")
        p.add_argument("--lr", type=float, default=0.1)
        p.add_argument("--rounds", type=int, default=1000)
        p.add_argument("--views", nargs="+", default=list(VIEWS),
                       help="retrieval/feature views; '<view>_<tag>' uses embeddings tagged <tag> (e.g. both_ft)")
    s1 = sp.choices["stage1"]
    s1.add_argument("--k", type=int, default=10)
    s1.add_argument("--train-s1", type=int, default=200_000)
    s1.add_argument("--prefilter", type=int, default=15)
    s1.add_argument("--chunk", type=int, default=200_000)
    s1.add_argument("--fit-pool", choices=["folds14", "fold0"], default="folds14",
                    help="fold0 = fold 0 minus dev (unseen by the E014 bi-encoder and the CE)")
    s1.add_argument("--max-s1", type=int, default=0, help="score only the first N S1 per split (smoke test)")
    s2 = sp.choices["stage2"]
    s2.add_argument("--train-s1", type=int, default=300_000)
    s2.add_argument("--fit-folds", type=int, nargs="+", default=[1, 2, 3, 4], help="folds whose S1s fit stage 2 (use 3 4 if the CE trained on folds 1-2)")
    s2.add_argument("--ce-dir", nargs="*", default=[], help="run dir(s) with train_ce/ and test_ce/ score parquets")
    s2.add_argument("--out", default="")
    s2.add_argument("--tag", default="", help="suffix for saved dev predictions (dev_stage2_<tag>.parquet)")
    s2.add_argument("--frames", default="", help="cache dir for the stage-2 design matrices (load if present, else save)")
    s2.add_argument("--lgb-params", default="", help='JSON LightGBM overrides, e.g. {"num_leaves": 255}')
    s2.add_argument("--unseen-threshold", type=float, default=0.0,
                    help="also write <out>_unseen: stricter threshold for S1 countries absent from train (same run)")
    s2.add_argument("--prune-eps", type=float, default=0.0,
                    help="final candidate set: keep pairs with stage-1 prob >= eps (e.g. 0.003 -> ~6.4 per S1)")
    s2.add_argument("--prune-ce-dir", default="", help="CE scores for the OR-rule candidate filter (e.g. runs/E016-ce)")
    s2.add_argument("--prune-ce", type=float, default=0.0, help="keep a pair if stage-1 prob >= --prune-eps OR CE >= this")
    s2.add_argument("--prune-unseen-ce-dir", default="", help="E037: CE adapted to countries absent from train (e.g. runs/E030-ce)")
    s2.add_argument("--prune-unseen-ce", type=float, default=0.0, help="E037: also keep unseen-country pairs with that CE >= this")
    s2.add_argument("--comp-keep", type=float, default=1.0,
                    help="E025: fraction of non-fit/dev train S1s kept as competitors (test-like density, e.g. 0.78)")
    s2.add_argument("--norm2", action="store_true", help="E024: add v2-normalised name/address features (src/er_norm2.py)")
    s2.add_argument("--comp-cols", nargs="*", default=[],
                    help="stage-1 columns for extra full-population competition features, e.g. name_full_tset")
    s2.add_argument("--stage3", action="store_true",
                    help="refit once with anchors re-picked from stage-2 probs; the test run then uses stage 3")
    a = ap.parse_args()
    {"stage1": cmd_stage1, "stage2": cmd_stage2}[a.cmd](a)


if __name__ == "__main__":
    main()
