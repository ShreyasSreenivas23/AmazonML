"""Competition metrics. Implement the official formula EXACTLY on Day 1 and unit-test it.

Every metric: fn(y_true, y_pred) -> float. Registry records direction so CV loops,
heartbeat, and early stopping never get "higher vs lower is better" wrong.

    from src.metrics import get_metric
    fn, higher_is_better = get_metric("smape")

Edge cases (zeros, empty strings, NaN) are where leaderboard formulas differ from
textbook ones — copy the organisers' definition, then add a test for each edge case.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

MetricFn = Callable[..., float]
_REGISTRY: dict[str, tuple[MetricFn, bool]] = {}


def register(name: str, higher_is_better: bool):
    """Decorator registering a metric function by name."""
    def deco(fn: MetricFn) -> MetricFn:
        """Register fn in the metric registry and return it unchanged."""
        _REGISTRY[name] = (fn, higher_is_better)
        return fn

    return deco


def get_metric(name: str) -> tuple[MetricFn, bool]:
    """Look up a registered metric by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    """Names of all registered metrics."""
    return sorted(_REGISTRY)


def _arr(x) -> np.ndarray:
    """Convert input to a float numpy array."""
    return np.asarray(x, dtype=np.float64)


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate that y_true and y_pred have the same length and are non-empty."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if not np.all(np.isfinite(y_pred)):
        raise ValueError("y_pred contains NaN/inf")


# ---------------------------------------------------------------- regression
@register("smape", higher_is_better=False)
def smape(y_true, y_pred) -> float:
    """SMAPE in percent, range [0, 200]. 0/0 terms count as 0 error (2025 challenge style).

    SMAPE = 100/n * sum(|F - A| / ((|A| + |F|) / 2))
    Verify zero-handling against the official definition on Day 1.
    """
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    diff = np.abs(f - a)
    ratio = np.divide(diff, denom, out=np.zeros_like(diff), where=denom != 0)
    return float(100.0 * ratio.mean())


@register("rmse", higher_is_better=False)
def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    return float(np.sqrt(np.mean((a - f) ** 2)))


@register("mae", higher_is_better=False)
def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    return float(np.mean(np.abs(a - f)))


@register("rmsle", higher_is_better=False)
def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error (log1p; negatives rejected)."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    if (a < 0).any() or (f < 0).any():
        raise ValueError("rmsle requires non-negative values")
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(f)) ** 2)))


@register("mape", higher_is_better=False)
def mape(y_true, y_pred) -> float:
    """MAPE in percent; rows with y_true == 0 are excluded (check official handling!)."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    m = a != 0
    return float(100.0 * np.mean(np.abs((a[m] - f[m]) / a[m])))


@register("r2", higher_is_better=True)
def r2(y_true, y_pred) -> float:
    """Coefficient of determination R^2."""
    a, f = _arr(y_true), _arr(y_pred)
    _check(a, f)
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


# ------------------------------------------------------------ classification
@register("accuracy", higher_is_better=True)
def accuracy(y_true, y_pred) -> float:
    """Classification accuracy."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@register("f1_macro", higher_is_better=True)
def f1_macro(y_true, y_pred) -> float:
    """Macro-averaged F1 over classes."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro"))


@register("f1_micro", higher_is_better=True)
def f1_micro(y_true, y_pred) -> float:
    """Micro-averaged F1."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="micro"))


@register("auc", higher_is_better=True)
def auc(y_true, y_score) -> float:
    """ROC AUC for binary labels."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


@register("logloss", higher_is_better=False)
def logloss(y_true, y_prob) -> float:
    """Binary log loss with clipped probabilities."""
    from sklearn.metrics import log_loss

    return float(log_loss(y_true, np.clip(_arr(y_prob), 1e-15, 1 - 1e-15)))


# ---------------------------------------------------------------- extraction
@register("extraction_f1", higher_is_better=True)
def extraction_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """2024-style entity-extraction F1 on exact string match; "" means "no prediction".

    TP: pred!="" & gt!="" & pred==gt     FP: pred!="" & (gt=="" or pred!=gt)
    FN: pred=="" & gt!=""                TN: pred=="" & gt==""
    """
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    tp = fp = fn = 0
    for gt, out in zip(y_true, y_pred):
        gt = "" if gt is None or (isinstance(gt, float) and np.isnan(gt)) else str(gt).strip()
        out = "" if out is None or (isinstance(out, float) and np.isnan(out)) else str(out).strip()
        if out and gt and out == gt:
            tp += 1
        elif out:
            fp += 1
        elif gt:
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * p * r / (p + r)) if p + r else 0.0


# ------------------------------------------------------------------- ranking
@register("ndcg", higher_is_better=True)
def ndcg(y_true, y_score, k: int | None = None) -> float:
    """y_true/y_score: 2D arrays (n_queries, n_items) of relevance / scores."""
    from sklearn.metrics import ndcg_score

    return float(ndcg_score(np.atleast_2d(y_true), np.atleast_2d(y_score), k=k))


@register("map_at_k", higher_is_better=True)
def map_at_k(actual: Sequence[Sequence], predicted: Sequence[Sequence], k: int = 10) -> float:
    """Mean average precision@k over queries (Kaggle definition)."""
    scores = []
    for act, pred in zip(actual, predicted):
        act = set(act)
        if not act:
            continue
        hits, s = 0, 0.0
        for i, p in enumerate(list(pred)[:k]):
            if p in act and p not in list(pred)[:i]:
                hits += 1
                s += hits / (i + 1)
        scores.append(s / min(len(act), k))
    return float(np.mean(scores)) if scores else 0.0


# ------------------------------------------------------- entity resolution (2026)
def parse_id_list(cell) -> set[str]:
    """'S2-1,S3-4' -> {'S2-1','S3-4'}; empty/NaN/None -> set()."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    return {t.strip() for t in str(cell).split(",") if t.strip()}


def entity_fbeta(true: set, pred: set, beta: float = 0.5) -> float:
    """Per-Source-1-entity F_beta exactly as the 2026 statement defines it.

    both empty -> 1.0 (correct singleton); exactly one empty -> 0.0; no overlap -> 0.0.
    """
    if not true and not pred:
        return 1.0
    tp = len(true & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


@register("er_f05", higher_is_better=True)
def er_fbeta_macro(y_true: dict, y_pred: dict, beta: float = 0.5) -> float:
    """Macro F_0.5 over ALL Source 1 entities in y_true (keys = S1 ids, values = sets of S2/S3 ids).

    S1 ids missing from y_pred count as empty predictions. Extra keys in y_pred are ignored
    (the portal would reject them; src/er_submission.py catches that separately).
    """
    if not y_true:
        raise ValueError("y_true is empty")
    return float(np.mean([entity_fbeta(set(t), set(y_pred.get(k, ())), beta) for k, t in y_true.items()]))
