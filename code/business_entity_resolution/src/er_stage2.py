"""Second-stage "cluster" features built from first-stage pair probabilities.

Diagnosis (E003): 74% of missed matches were retrieved but rejected. All true matches of one S1 are copies of the
same business, so a hard true match tends to resemble the S1's *confident* matches (anchors) more than the S1 text.

    F = cluster_features(P, right, emb_right)   # P: [s1_id, cand_id, prob] with OUT-OF-FOLD probs for training rows
    -> float32 features aligned with P's row order

Leakage rule: training rows must use out-of-fold first-stage probabilities (never in-fold), so the second stage learns
from the same kind of noisy probabilities it will see on dev/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


def _pairwise(a, b, scorer) -> np.ndarray:
    """Element-wise rapidfuzz scorer over aligned string arrays, as float32 in [0, 1]."""
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / 100.0).astype(np.float32)


def competition_features(P: pd.DataFrame, col: str = "prob") -> dict[str, np.ndarray]:
    """For each pair: the best `col` value its candidate record has with any OTHER S1, and the margin over it.
    Needs every S1 that retrieved the record, so compute it over all pairs, not per chunk of S1s.

    col="prob" keeps the original names (s2_other_s1_best, s2_margin_vs_other_s1); other columns (e.g. a name
    similarity) give comp_<col>_other_best / comp_<col>_margin. Error analysis (E015 dev): 85% of the loss is true
    empty-address matches rejected because near-identical empty-address records of OTHER S1s compete; whether this
    S1 is the record's best NAME match among all S1s is the missing signal (prob is low for all of them)."""
    v = P[col].to_numpy(np.float32)
    gc = P.groupby("cand_id", sort=False)[col]
    top1 = gc.transform("max").to_numpy(np.float32)
    r = gc.rank(ascending=False, method="first").to_numpy()
    second = P.loc[r == 2, ["cand_id", col]].set_index("cand_id")[col]
    top2 = P.cand_id.map(second).fillna(0.0).to_numpy(np.float32)
    other_best = np.where(r == 1, top2, top1).astype(np.float32)
    if col == "prob":
        return {"s2_other_s1_best": other_best, "s2_margin_vs_other_s1": (v - other_best).astype(np.float32)}
    return {f"comp_{col}_other_best": other_best, f"comp_{col}_margin": (v - other_best).astype(np.float32)}


def cluster_features(P: pd.DataFrame, right: pd.DataFrame, emb_right: dict, n_anchors: int = 3,
                     anchor_min: float = 0.5, step: int = 1_000_000) -> pd.DataFrame:
    """Per-pair features from the S1's other candidates and the record's other S1s."""
    P = P.reset_index(drop=True)
    prob = P.prob.to_numpy(np.float32)
    g1 = P.groupby("s1_id", sort=False).prob
    out = {
        "s2_prob": prob,
        "s2_rank_in_s1": g1.rank(ascending=False, method="first").to_numpy(np.float32),
        "s2_gap_to_max_s1": (g1.transform("max") - P.prob).to_numpy(np.float32),
        "s2_n_conf50_s1": P.assign(_c=prob >= 0.5).groupby("s1_id", sort=False)._c.transform("sum").to_numpy(np.float32),
        "s2_n_conf90_s1": P.assign(_c=prob >= 0.9).groupby("s1_id", sort=False)._c.transform("sum").to_numpy(np.float32),
        "s2_sum_prob_s1": g1.transform("sum").to_numpy(np.float32),
    }
    out.update(competition_features(P))

    # anchors: top-n confident candidates per S1; compare every candidate with its S1's anchors (excluding itself)
    rank = out["s2_rank_in_s1"]
    A = P.loc[(rank <= n_anchors) & (prob >= anchor_min), ["s1_id", "cand_id", "prob"]]
    A = A.rename(columns={"cand_id": "anchor", "prob": "anchor_prob"})
    J = P[["s1_id", "cand_id"]].reset_index().merge(A, on="s1_id", how="inner")
    J = J[J.cand_id != J.anchor].reset_index(drop=True)
    ci = right.index.get_indexer(J.cand_id)
    ai = right.index.get_indexer(J.anchor)
    feats = {}
    for view, E in emb_right.items():
        cos = np.empty(len(J), np.float32)
        for s in range(0, len(J), step):
            a = np.asarray(E[ci[s:s + step]], dtype=np.float32)
            b = np.asarray(E[ai[s:s + step]], dtype=np.float32)
            cos[s:s + step] = np.einsum("ij,ij->i", a, b)
        feats[f"sib_cos_{view}"] = cos
    cn, an = right.name_core.to_numpy(), right.addr_norm.to_numpy()
    feats["sib_name_tset"] = _pairwise(cn[ci], cn[ai], fuzz.token_set_ratio)
    feats["sib_addr_tset"] = _pairwise(an[ci], an[ai], fuzz.token_set_ratio)
    Jf = pd.DataFrame(feats)
    Jf["row"] = J["index"].to_numpy()
    Jf["w"] = J.anchor_prob.to_numpy(np.float32)
    agg = Jf.groupby("row")
    n = len(P)
    for c in feats:
        mx = np.full(n, np.nan, np.float32)
        wm = np.full(n, np.nan, np.float32)
        m = agg[c].max()
        mx[m.index.to_numpy()] = m.to_numpy()
        num = (Jf[c] * Jf.w).groupby(Jf.row).sum()
        den = Jf.w.groupby(Jf.row).sum()
        wm[num.index.to_numpy()] = (num / den).to_numpy()
        out[f"{c}_max"] = mx
        out[f"{c}_wmean"] = wm
    cnt = np.zeros(n, np.float32)
    k = agg.size()
    cnt[k.index.to_numpy()] = k.to_numpy()
    out["sib_n_anchors"] = cnt
    return pd.DataFrame(out)
