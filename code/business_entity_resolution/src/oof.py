"""Out-of-fold (OOF) + test prediction storage. Every model saves these from Day 1.

Layout (gitignored, shared via the team drive — see docs/TEAM.md):
    oof/<exp_id>/oof.parquet    id, fold, pred[_0.._k]   (one row per train id)
    oof/<exp_id>/test.parquet   id, pred[_0.._k]         (fold-averaged or full-fit)
    oof/<exp_id>/meta.json      metric, cv, per-fold cv, target transform, notes

OOF preds must be in the ORIGINAL target space (inverse-transform first), so any
two experiments can be blended/stacked directly.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_env


def oof_root() -> Path:
    """Root folder for out-of-fold prediction files (OOF_DIR)."""
    load_env()
    p = Path(os.environ.get("OOF_DIR") or "oof")
    return p if p.is_absolute() else REPO_ROOT / p


def _pred_frame(pred: np.ndarray) -> pd.DataFrame:
    """Predictions as a DataFrame with an id column."""
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pd.DataFrame({"pred": pred})
    return pd.DataFrame({f"pred_{j}": pred[:, j] for j in range(pred.shape[1])})


def save_oof(
    exp_id: str,
    train_ids: Sequence,
    oof_pred: np.ndarray,
    folds: Sequence[int],
    test_ids: Sequence | None = None,
    test_pred: np.ndarray | None = None,
    metric: str | None = None,
    cv: float | None = None,
    fold_scores: Sequence[float] | None = None,
    root: str | Path | None = None,
    **meta,
) -> Path:
    """Save OOF and test predictions of one experiment for later blending."""
    oof_pred = np.asarray(oof_pred)
    if len(train_ids) != len(oof_pred) or len(folds) != len(oof_pred):
        raise ValueError("train_ids, oof_pred, folds must have equal length")
    if not np.all(np.isfinite(oof_pred.astype(np.float64, copy=False))):
        raise ValueError("oof_pred has NaN/inf — some fold did not predict?")
    d = Path(root or oof_root()) / exp_id
    d.mkdir(parents=True, exist_ok=True)
    oof = pd.concat([pd.DataFrame({"id": list(train_ids), "fold": list(folds)}), _pred_frame(oof_pred)], axis=1)
    oof.to_parquet(d / "oof.parquet", index=False)
    if test_pred is not None:
        if test_ids is None or len(test_ids) != len(test_pred):
            raise ValueError("test_ids must match test_pred length")
        pd.concat([pd.DataFrame({"id": list(test_ids)}), _pred_frame(test_pred)], axis=1).to_parquet(
            d / "test.parquet", index=False
        )
    info = {
        "exp_id": exp_id,
        "metric": metric,
        "cv": cv,
        "fold_scores": list(map(float, fold_scores)) if fold_scores is not None else None,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    (d / "meta.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return d


def load_oof(exp_id: str, root: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Load an experiment's saved OOF and test predictions."""
    d = Path(root or oof_root()) / exp_id
    oof = pd.read_parquet(d / "oof.parquet")
    test = pd.read_parquet(d / "test.parquet") if (d / "test.parquet").exists() else None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return oof, test, meta


def stack_oofs(exp_ids: Sequence[str], root: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align several experiments' OOF/test preds by id -> wide frames (one column per exp).

    Raises if ids or folds disagree — mismatched folds silently leak in stacking.
    """
    oofs, tests, ref, test_ref = [], [], None, None
    for e in exp_ids:
        oof, test, _ = load_oof(e, root)
        oof = oof.sort_values("id").reset_index(drop=True)
        if ref is None:
            ref = oof[["id", "fold"]]
        elif not (oof["id"].equals(ref["id"]) and oof["fold"].equals(ref["fold"])):
            raise ValueError(f"{e}: ids/folds differ from {exp_ids[0]} — regenerate with shared folds")
        pcols = [c for c in oof.columns if c.startswith("pred")]
        oofs.append(oof[pcols].add_prefix(f"{e}__"))
        if test is not None:
            test = test.sort_values("id").reset_index(drop=True)
            if test_ref is None:
                test_ref = test[["id"]]
            elif not test["id"].equals(test_ref["id"]):
                raise ValueError(f"{e}: test ids differ from other experiments")
            tests.append(test[[c for c in test.columns if c.startswith("pred")]].add_prefix(f"{e}__"))
    X_oof = pd.concat([ref] + oofs, axis=1)
    X_test = pd.concat([test_ref] + tests, axis=1) if tests and len(tests) == len(exp_ids) else None
    return X_oof, X_test
