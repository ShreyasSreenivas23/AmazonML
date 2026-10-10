"""CV splitter. Choose the scheme that mimics train->test (see docs/PLAYBOOK_72H.md §CV).

    folds = make_folds(df, n_splits=5, method="stratified_reg", target="y", seed=42)
    save_folds(folds, "data/folds.csv")   # every model uses the SAME folds file

Returns a DataFrame [id_col, "fold"]. Fixed, shared folds are what make OOF
predictions from different people/models stackable without leakage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

METHODS = ("kfold", "stratified", "stratified_reg", "group", "stratified_group", "time")


def _reg_bins(y: pd.Series, n_bins: int) -> np.ndarray:
    """Quantile-bin a continuous target so StratifiedKFold can balance it."""
    return pd.qcut(y.rank(method="first"), q=n_bins, labels=False).to_numpy()


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    method: str = "kfold",
    target: str | None = None,
    group: str | None = None,
    time_col: str | None = None,
    id_col: str = "id",
    seed: int = 42,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Assign k folds (grouped or stratified as configured) with a fixed seed; returns a fold column."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    n = len(df)
    fold = np.full(n, -1, dtype=int)
    idx = np.arange(n)

    if method == "kfold":
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(idx)
    elif method == "stratified":
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, df[target])
    elif method == "stratified_reg":
        bins = _reg_bins(df[target], n_bins)
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(idx, bins)
    elif method == "group":
        # GroupKFold is deterministic; shuffle group ids for seed variation.
        g = df[group].astype("category").cat.codes.to_numpy()
        perm = np.random.RandomState(seed).permutation(g.max() + 1)
        splits = GroupKFold(n_splits).split(idx, groups=perm[g])
    elif method == "stratified_group":
        y = df[target]
        y = _reg_bins(y, n_bins) if pd.api.types.is_float_dtype(y) else y
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(
            idx, y, groups=df[group]
        )
    else:  # time: expanding-window style; fold k = k-th time chunk (use fold>0 as val)
        order = np.argsort(df[time_col].to_numpy(), kind="stable")
        for k, chunk in enumerate(np.array_split(order, n_splits)):
            fold[chunk] = k
        splits = None

    if splits is not None:
        for k, (_, val_idx) in enumerate(splits):
            fold[val_idx] = k
    assert (fold >= 0).all(), "some rows not assigned a fold"
    ids = df[id_col].to_numpy() if id_col in df.columns else idx
    return pd.DataFrame({id_col: ids, "fold": fold})


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    """Persist the shared folds file so every model uses identical splits."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)


def load_folds(path: str | Path) -> pd.DataFrame:
    """Load the shared folds file written by save_folds."""
    return pd.read_csv(path)


def iter_folds(folds: pd.DataFrame):
    """Yield (k, train_idx, val_idx) as positional indices."""
    f = folds["fold"].to_numpy()
    for k in sorted(np.unique(f)):
        yield k, np.where(f != k)[0], np.where(f == k)[0]
