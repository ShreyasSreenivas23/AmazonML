"""Decision layer: pair probabilities -> final match sets per S1. Contract: docs/WORKERS.md.

    decide(pairs_with_prob, threshold=0.5, assign=True, top1_fallback=None) -> {s1_id: set(cand_ids)}
    decide_expected_f(pairs_with_prob, beta=0.5, assign=True)              -> {s1_id: set(cand_ids)}
    tune_threshold(pairs_with_prob, y_true, s1_ids, grid)                   -> (best_t, best_score, table)

Both strategies are compared on held-out per-entity F0.5 (docs/DECISIONS.md, review point 5); neither is assumed.
`assign=True` enforces the train-set fact that each S2/S3 record belongs to at most one S1 (switchable, ablated).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import er_fbeta_macro


def assign_records(p: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each cand_id, only its highest-probability S1 (ties -> first)."""
    return p.sort_values("prob", ascending=False, kind="stable").drop_duplicates("cand_id")


def decide(p: pd.DataFrame, threshold: float = 0.5, assign: bool = True,
           top1_fallback: float | None = None) -> dict[str, set[str]]:
    """Global threshold. Optional fallback: if an S1 has nothing above threshold, keep its best candidate when
    that candidate's prob >= top1_fallback (a lower bar; helps because only ~5.6% of S1 are true singletons)."""
    q = assign_records(p) if assign else p
    keep = q[q.prob >= threshold]
    out: dict[str, set[str]] = {}
    for s1, c in zip(keep.s1_id.to_numpy(), keep.cand_id.to_numpy()):
        out.setdefault(s1, set()).add(c)
    if top1_fallback is not None:
        best = q.sort_values("prob", ascending=False, kind="stable").drop_duplicates("s1_id")
        best = best[(best.prob >= top1_fallback) & ~best.s1_id.isin(out.keys())]
        for s1, c in zip(best.s1_id.to_numpy(), best.cand_id.to_numpy()):
            out[s1] = {c}
    return out


def expected_fbeta_prefix(probs: np.ndarray, beta: float = 0.5, n_samples: int = 256, seed: int = 0) -> int:
    """Best prefix length (0 = empty set) of candidates sorted by prob, maximising Monte-Carlo expected F_beta,
    treating candidate labels as independent Bernoulli(prob). Exact enough for <= ~30 candidates."""
    probs = np.asarray(probs, dtype=np.float64)
    if len(probs) == 0:
        return 0
    rng = np.random.default_rng(seed)
    y = rng.random((n_samples, len(probs))) < probs        # sampled truth for each candidate
    n_true = y.sum(1)
    b2 = beta * beta
    best_k, best_v = 0, float(np.mean(n_true == 0))         # empty prediction: 1 only if no true match
    tp = np.zeros(n_samples)
    for k in range(1, len(probs) + 1):
        tp += y[:, k - 1]
        prec = tp / k
        rec = np.divide(tp, n_true, out=np.zeros(n_samples), where=n_true > 0)
        f = np.divide((1 + b2) * prec * rec, b2 * prec + rec, out=np.zeros(n_samples), where=(prec + rec) > 0)
        v = float(f.mean())
        if v > best_v:
            best_k, best_v = k, v
    return best_k


def decide_expected_f(p: pd.DataFrame, beta: float = 0.5, assign: bool = True, max_cands: int = 20,
                      min_prob: float = 0.01) -> dict[str, set[str]]:
    """Per-S1 expected-F_beta-optimal subset over calibrated probabilities."""
    q = assign_records(p) if assign else p
    q = q[q.prob >= min_prob].sort_values(["s1_id", "prob"], ascending=[True, False], kind="stable")
    out: dict[str, set[str]] = {}
    for s1, g in q.groupby("s1_id", sort=False):
        pr = g.prob.to_numpy()[:max_cands]
        k = expected_fbeta_prefix(pr, beta)
        if k:
            out[s1] = set(g.cand_id.to_numpy()[:k])
    return out


def tune_threshold(p: pd.DataFrame, y_true: dict[str, set[str]], grid=None, assign: bool = True,
                   top1_fallback: float | None = None) -> tuple[float, float, pd.DataFrame]:
    """Grid-search the global threshold for macro F0.5 over ALL S1 in y_true (entities without candidates count)."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    rows = [{"threshold": t, "f05": er_fbeta_macro(y_true, decide(p, t, assign, top1_fallback))} for t in grid]
    tab = pd.DataFrame(rows)
    best = tab.loc[tab.f05.idxmax()]
    return float(best.threshold), float(best.f05), tab
