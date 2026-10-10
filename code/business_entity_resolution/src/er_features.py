"""Pair features for the matcher. Contract: docs/WORKERS.md.

    X = build_features(pairs, left, right, emb_left, emb_right)
      pairs      DataFrame[s1_id, cand_id, (optional retrieval cols: rank_<view>)]
      left/right normalised frames (src.er_normalize.normalize_frame output) indexed by entity_id
      emb_*      {view: float16 matrix} aligned with left/right row order
    -> DataFrame[s1_id, cand_id, <float32 features>] in the same row order as pairs.

Country-agnostic by design: no feature encodes *which* country a record is from (France is unseen in train).
Feature groups register themselves with @feature_group so new groups can be added in parallel (WORKERS.md).
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

FEATURE_GROUPS: dict[str, Callable] = {}


def feature_group(name: str):
    """Decorator registering a feature-group function under a name."""
    def deco(fn):
        """Register fn in FEATURE_GROUPS and return it unchanged."""
        FEATURE_GROUPS[name] = fn
        return fn
    return deco


def _pairwise(a: np.ndarray, b: np.ndarray, scorer) -> np.ndarray:
    """Element-wise scorer(a[i], b[i]) over all 32 threads, as float32 in [0, 1]."""
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / 100.0).astype(np.float32)


@feature_group("embed")
def f_embed(ctx: dict) -> dict:
    """Cosine similarity per embedding view (computed for every pair, not only for retrieved ones)."""
    out = {}
    li, ri = ctx["li"], ctx["ri"]
    step = 1_000_000  # chunked: gathering all pairs at once needs ~23 GB per side per view at 15M pairs
    for view, (el, er) in ctx["emb"].items():
        cos = np.empty(len(li), dtype=np.float32)
        for s in range(0, len(li), step):
            a = np.asarray(el[li[s:s + step]], dtype=np.float32)
            b = np.asarray(er[ri[s:s + step]], dtype=np.float32)
            cos[s:s + step] = np.einsum("ij,ij->i", a, b)
        out[f"cos_{view}"] = cos
    return out


@feature_group("name")
def f_name(ctx: dict) -> dict:
    """Name similarities: ratio, token set/sort, partial, Jaro-Winkler, exact core-name match, lengths."""
    L, R = ctx["L"], ctx["R"]
    a, b = L["name_core"], R["name_core"]
    an, bn = L["name_norm"], R["name_norm"]
    return {
        "name_ratio": _pairwise(a, b, fuzz.ratio),
        "name_tset": _pairwise(a, b, fuzz.token_set_ratio),
        "name_tsort": _pairwise(a, b, fuzz.token_sort_ratio),
        "name_partial": _pairwise(a, b, fuzz.partial_ratio),
        "name_jw": (process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1)).astype(np.float32),
        "name_full_tset": _pairwise(an, bn, fuzz.token_set_ratio),
        "name_exact_core": (a == b).astype(np.float32),
        "name_len_l": np.char.str_len(a.astype(str)).astype(np.float32),
        "name_len_r": np.char.str_len(b.astype(str)).astype(np.float32),
    }


@feature_group("address")
def f_address(ctx: dict) -> dict:
    """Address similarities plus number agreement (Jaccard, first number) and postcode agree/conflict."""
    L, R = ctx["L"], ctx["R"]
    a, b = L["addr_norm"], R["addr_norm"]
    empty_r = (np.char.str_len(b.astype(str)) == 0)
    feats = {
        "addr_tset": _pairwise(a, b, fuzz.token_set_ratio),
        "addr_partial": _pairwise(a, b, fuzz.partial_ratio),
        "addr_tsort": _pairwise(a, b, fuzz.token_sort_ratio),
        "addr_empty_r": empty_r.astype(np.float32),
    }
    # numbers: Jaccard of digit-run sets, first number equal, postcode agree/conflict
    na, nb = L["addr_nums"], R["addr_nums"]
    jac, first_eq, n_common = [], [], []
    for x, y in zip(na, nb):
        sx, sy = set(x.split()), set(y.split())
        u = len(sx | sy)
        c = len(sx & sy)
        jac.append(c / u if u else np.nan)
        n_common.append(c)
        fx, fy = x.split(" ", 1)[0], y.split(" ", 1)[0]
        first_eq.append(float(fx == fy) if fx and fy else np.nan)
    pa, pb = L["postcode"], R["postcode"]
    both = (pa != "") & (pb != "")
    feats.update({
        "num_jaccard": np.array(jac, np.float32),
        "num_common": np.array(n_common, np.float32),
        "num_first_eq": np.array(first_eq, np.float32),
        "post_eq": np.where(both, (pa == pb).astype(np.float32), np.nan).astype(np.float32),
    })
    return feats


@feature_group("context")
def f_context(ctx: dict, key: str = "cos_both") -> dict:
    """Where this pair sits among the S1's candidates and among the record's S1s (mutual rank)."""
    df = pd.DataFrame({"s1": ctx["pairs"]["s1_id"].to_numpy(), "c": ctx["pairs"]["cand_id"].to_numpy(),
                       "src": ctx["pairs"]["cand_id"].str[:2].to_numpy(), "v": ctx["base"][key]})
    g1 = df.groupby(["s1", "src"])["v"]
    g2 = df.groupby("c")["v"]
    return {
        "ctx_rank_in_s1": g1.rank(ascending=False, method="min").to_numpy(np.float32),
        "ctx_gap_to_best_s1": (g1.transform("max") - df.v).to_numpy(np.float32),
        "ctx_n_cands_s1": g1.transform("size").to_numpy(np.float32),
        "ctx_rank_in_cand": g2.rank(ascending=False, method="min").to_numpy(np.float32),
        "ctx_gap_to_best_cand": (g2.transform("max") - df.v).to_numpy(np.float32),
        "ctx_n_s1_for_cand": g2.transform("size").to_numpy(np.float32),
        "is_s3": (df.src == "S3").to_numpy(np.float32),
    }


def build_features(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, emb: dict | None = None,
                   groups: list[str] | None = None) -> pd.DataFrame:
    """Compute all (or selected) feature groups for the pairs; returns s1_id, cand_id + float32 features."""
    li = left.index.get_indexer(pairs["s1_id"])
    ri = right.index.get_indexer(pairs["cand_id"])
    if (li < 0).any() or (ri < 0).any():
        raise KeyError("pairs reference ids missing from left/right frames")
    cols = ["name_norm", "name_core", "addr_norm", "addr_nums", "postcode"]
    ctx = {
        "pairs": pairs, "li": li, "ri": ri, "emb": emb or {},
        "L": {c: left[c].to_numpy()[li] for c in cols},
        "R": {c: right[c].to_numpy()[ri] for c in cols},
        "base": {},
    }
    out: dict[str, np.ndarray] = {}
    order = groups or [g for g in FEATURE_GROUPS if g != "context"] + ["context"]
    for g in order:
        if g == "context" and "cos_both" not in out:
            continue  # context needs the combined-view cosine
        feats = FEATURE_GROUPS[g](ctx)
        out.update(feats)
        ctx["base"].update(feats)
    for c in pairs.columns:
        if c.startswith("rank_"):
            out[c] = pairs[c].to_numpy(np.float32)
    X = pd.DataFrame(out)
    X.insert(0, "cand_id", pairs["cand_id"].to_numpy())
    X.insert(0, "s1_id", pairs["s1_id"].to_numpy())
    return X
