"""Pair scorer: LightGBM (MIT) with grouped out-of-fold predictions. Contract: docs/WORKERS.md.

    oof, models = train_oof(X, y, groups, cfg)      # groups = S1 fold ids, so an S1's pairs never straddle folds
    prob = predict(models, X_test)                  # mean over fold models
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "num_threads": 32, "verbose": -1, "seed": 42,
}


def feature_cols(X: pd.DataFrame) -> list[str]:
    """Model feature columns: everything except ids and the label."""
    return [c for c in X.columns if c not in ("s1_id", "cand_id", "y")]


def train_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, params: dict | None = None,
              num_boost_round: int = 2000, early_stopping: int = 100) -> tuple[np.ndarray, list[lgb.Booster]]:
    """One model per distinct group value (fold); returns OOF probabilities and the fitted boosters."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    cols = feature_cols(X)
    oof = np.full(len(X), np.nan, dtype=np.float64)
    models = []
    for g in np.unique(groups):
        va = groups == g
        dtr = lgb.Dataset(X.loc[~va, cols], label=y[~va], free_raw_data=True)
        dva = lgb.Dataset(X.loc[va, cols], label=y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X.loc[va, cols], num_iteration=m.best_iteration)
        models.append(m)
    return oof, models


def predict(models: list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """Mean probability over the fold models (each at its best iteration)."""
    cols = models[0].feature_name()
    return np.mean([m.predict(X[cols], num_iteration=m.best_iteration) for m in models], axis=0)


def importance(models: list[lgb.Booster]) -> pd.Series:
    """Gain importance summed over fold models, normalised to 1."""
    imp = sum(pd.Series(m.feature_importance("gain"), index=m.feature_name()) for m in models)
    return (imp / imp.sum()).sort_values(ascending=False)
