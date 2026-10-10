"""Submission writer + validator. A format error wastes a daily submission — always validate.

    write_submission(ids, preds, "submissions/E007.csv", id_col="sample_id",
                     target_col="price", sample="data/sample_test_out.csv")

CLI (run before EVERY upload; exit code 1 on any failure):
    python -m src.submission validate submissions/E007.csv --sample data/sample_test_out.csv
    python -m src.submission validate preds.csv --test data/test.csv --id-col sample_id --target-col price --nonneg
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Report:
    """Validation result: errors, warnings and info lines."""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when there are no errors."""
        return not self.errors

    def __str__(self) -> str:
        """Human-readable PASS/FAIL report."""
        lines = [f"[{'PASS' if self.ok else 'FAIL'}] submission validation"]
        lines += [f"  ERROR: {e}" for e in self.errors]
        lines += [f"  WARN:  {w}" for w in self.warnings]
        lines += [f"  info:  {i}" for i in self.info]
        return "\n".join(lines)


def _read(path: str | Path) -> pd.DataFrame:
    # Read ids as strings so "007" vs 7 mismatches surface instead of silently coercing.
    """Read a submission CSV/TSV as strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_submission(
    path: str | Path,
    sample: str | Path | None = None,
    test: str | Path | None = None,
    id_col: str | None = None,
    target_cols: Sequence[str] | None = None,
    nonneg: bool = False,
    allow_reorder: bool = False,
) -> Report:
    """Generic submission check against a sample file (columns, ids, NaNs, value ranges)."""
    r = Report()
    path = Path(path)
    if not path.exists():
        r.errors.append(f"file not found: {path}")
        return r
    sub = _read(path)
    r.info.append(f"{path.name}: {len(sub)} rows, columns={list(sub.columns)}")

    ref_ids = None
    if sample is not None:
        smp = _read(sample)
        if list(sub.columns) != list(smp.columns):
            r.errors.append(f"columns {list(sub.columns)} != sample {list(smp.columns)}")
        id_col = id_col or smp.columns[0]
        target_cols = target_cols or [c for c in smp.columns if c != id_col]
        ref_ids = smp[id_col] if id_col in smp.columns else None
    elif test is not None:
        if id_col is None:
            r.errors.append("--id-col required with --test")
            return r
        ref_ids = _read(test)[id_col]
    id_col = id_col or sub.columns[0]
    target_cols = list(target_cols or [c for c in sub.columns if c != id_col])

    if any(c.startswith("Unnamed") for c in sub.columns):
        r.errors.append("found 'Unnamed' column — did you write the DataFrame index?")
    if id_col not in sub.columns:
        r.errors.append(f"id column {id_col!r} missing")
        return r
    missing = [c for c in target_cols if c not in sub.columns]
    if missing:
        r.errors.append(f"target columns missing: {missing}")
        return r

    ids = sub[id_col]
    if ids.duplicated().any():
        r.errors.append(f"{int(ids.duplicated().sum())} duplicate ids")
    if (ids.str.strip() == "").any():
        r.errors.append("empty id values")

    if ref_ids is not None:
        if len(sub) != len(ref_ids):
            r.errors.append(f"row count {len(sub)} != expected {len(ref_ids)}")
        s_sub, s_ref = set(ids), set(ref_ids)
        if s_sub != s_ref:
            r.errors.append(
                f"id set mismatch: {len(s_ref - s_sub)} expected ids missing, "
                f"{len(s_sub - s_ref)} unexpected ids (e.g. {list(s_sub - s_ref)[:3]})"
            )
        elif not ids.reset_index(drop=True).equals(ref_ids.reset_index(drop=True)):
            msg = "row order differs from reference"
            (r.warnings if allow_reorder else r.errors).append(msg + ("" if allow_reorder else " (use --allow-reorder if grader joins by id)"))
    else:
        r.warnings.append("no --sample/--test given: row count and id alignment NOT checked")

    for c in target_cols:
        col = sub[c]
        blank = (col.str.strip() == "") | col.str.lower().isin(["nan", "none", "null", "inf", "-inf"])
        if blank.any():
            r.errors.append(f"{c}: {int(blank.sum())} empty/NaN/inf values")
        num = pd.to_numeric(col, errors="coerce")
        if num.notna().all():
            if not np.isfinite(num.to_numpy()).all():
                r.errors.append(f"{c}: non-finite values")
            if nonneg and (num < 0).any():
                r.errors.append(f"{c}: {int((num < 0).sum())} negative values")
            r.info.append(f"{c}: min={num.min():.6g} median={num.median():.6g} max={num.max():.6g} n_unique={num.nunique()}")
            if num.nunique() == 1:
                r.warnings.append(f"{c}: constant prediction")
        else:
            r.info.append(f"{c}: non-numeric (string) column; {col.nunique()} unique, e.g. {col.iloc[0]!r}")
    return r


def write_submission(
    ids: Sequence,
    preds: np.ndarray | Sequence,
    path: str | Path,
    id_col: str = "id",
    target_col: str = "target",
    sample: str | Path | None = None,
    float_format: str | None = None,
    nonneg: bool = False,
) -> Report:
    """Write CSV (ordered like `sample` if given), validate, raise on failure."""
    df = pd.DataFrame({id_col: list(ids), target_col: list(preds)})
    if sample is not None:
        smp = pd.read_csv(sample, dtype=str, keep_default_na=False)
        df[id_col] = df[id_col].astype(str)
        df = smp[[id_col]].merge(df, on=id_col, how="left")
        df = df[list(smp.columns)] if set(smp.columns) == set(df.columns) else df
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=float_format)
    rep = validate_submission(path, sample=sample, id_col=id_col, target_cols=[target_col], nonneg=nonneg)
    print(rep)
    if not rep.ok:
        raise ValueError(f"submission {path} failed validation")
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: validate a generic submission file."""
    ap = argparse.ArgumentParser(prog="python -m src.submission")
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate", help="validate a predictions file")
    v.add_argument("path")
    v.add_argument("--sample", help="official sample submission (preferred reference)")
    v.add_argument("--test", help="test file to take expected ids from")
    v.add_argument("--id-col")
    v.add_argument("--target-col", action="append", dest="target_cols")
    v.add_argument("--nonneg", action="store_true", help="fail on negative predictions")
    v.add_argument("--allow-reorder", action="store_true")
    a = ap.parse_args(argv)
    rep = validate_submission(a.path, a.sample, a.test, a.id_col, a.target_cols, a.nonneg, a.allow_reorder)
    print(rep)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
