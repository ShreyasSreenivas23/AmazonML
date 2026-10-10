"""Entity-resolution I/O: load sources + ground truth, write and validate the two output TSVs.

    from src.er_submission import load_sources, load_ground_truth, write_outputs
    s1, s2, s3 = load_sources("data/dataset/test")
    write_outputs(matches, candidates, s1["entity_id"], "output/")      # validates, raises on any error

CLI (run before EVERY upload, then also run the organisers' utils/validate_submission.py):
    python -m src.er_submission validate --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir data/dataset/test

Mirrors the rejection rules in docs/COMPETITION.md: one row per test S1 id, no duplicate rows, ID lists contain
only existing test S2-/S3- ids, no duplicates inside a list, and matches must be a subset of candidates.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from src.metrics import parse_id_list
from src.submission import Report

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


# ------------------------------------------------------------------ loading
def read_tsv(path: str | Path, usecols=None) -> pd.DataFrame:
    """Read an organiser TSV: tab-separated, everything as str, empty cells stay ''."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3,  # 3 = QUOTE_NONE
                       usecols=usecols)


def load_sources(split_dir: str | Path, usecols=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """split_dir contains <split>_source{1,2,3}.tsv (e.g. dataset/test/test_source1.tsv).
    Pass usecols=["entity_id"] when only IDs are needed (the validator) to keep memory low."""
    d = Path(split_dir)
    out = []
    for k in (1, 2, 3):
        files = sorted(d.glob(f"*_source{k}.tsv"))
        if len(files) != 1:
            raise FileNotFoundError(f"expected exactly one *_source{k}.tsv in {d}, found {files}")
        out.append(read_tsv(files[0], usecols))
    return out[0], out[1], out[2]


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    """Ground truth as {S1 id: set of matched ids}."""
    gt = read_tsv(path)
    return {r.source1_entity_id: parse_id_list(r.matched_entity_ids) for r in gt.itertuples(index=False)}


# ------------------------------------------------------------------ writing
def _fmt(ids: Iterable[str]) -> str:
    """Comma-join an ID list (empty string for no ids)."""
    return ",".join(sorted(set(ids)))


def write_outputs(matches: Mapping[str, Iterable[str]], candidates: Mapping[str, Iterable[str]],
                  s1_ids: Iterable[str], out_dir: str | Path, test_dir: str | Path | None = None) -> Report:
    """Write matching_results.tsv + candidate_pairs.tsv (one row per S1 id, in s1_ids order), then validate."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s1_ids = list(s1_ids)
    for name, header, data in (("matching_results.tsv", MATCH_HEADER, matches),
                               ("candidate_pairs.tsv", CAND_HEADER, candidates)):
        with open(out / name, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(header) + "\n")
            for s in s1_ids:
                f.write(f"{s}\t{_fmt(data.get(s, ()))}\n")
    rep = validate_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir=test_dir,
                           s1_ids=None if test_dir else s1_ids)
    print(rep)
    if not rep.ok:
        raise ValueError(f"outputs in {out} failed validation")
    return rep


# --------------------------------------------------------------- validating
def _parse_file(path: Path, header: tuple[str, str], r: Report) -> dict[str, list[str]] | None:
    """Strict line parser (no pandas) so we see exactly what the grader sees."""
    if not path.exists():
        r.errors.append(f"{path.name}: file not found")
        return None
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        r.errors.append(f"{path.name}: has a UTF-8 BOM; write with encoding='utf-8' (not utf-8-sig)")
    if b"\r\n" in raw:
        r.warnings.append(f"{path.name}: CRLF line endings; prefer '\\n'")
    lines = raw.decode("utf-8-sig", "replace").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines or tuple(lines[0].split("\t")) != header:
        r.errors.append(f"{path.name}: header must be {'<TAB>'.join(header)!r}, got {lines[0] if lines else ''!r}")
        return None
    rows: dict[str, list[str]] = {}
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) != 2:
            r.errors.append(f"{path.name}:{i}: expected 2 tab-separated fields, got {len(parts)}")
            continue
        s1, lst = parts
        if s1 in rows:
            r.errors.append(f"{path.name}:{i}: duplicate source1_entity_id {s1}")
        ids = [t for t in lst.split(",")] if lst else []
        if any(t != t.strip() or not t for t in ids) or '"' in lst:
            r.errors.append(f"{path.name}:{i}: ID list has spaces/empty items/quotes: {lst[:80]!r}")
        rows[s1] = [t.strip() for t in ids if t.strip()]
    return rows


def _check_lists(name: str, rows: dict[str, list[str]], valid_s1: set[str] | None, valid_other: set[str] | None,
                 r: Report) -> None:
    """Check one file's ID lists: known S2/S3 ids only, no self-matches, no duplicates."""
    if valid_s1 is not None:
        missing, extra = valid_s1 - rows.keys(), rows.keys() - valid_s1
        if missing:
            r.errors.append(f"{name}: {len(missing)} test S1 ids missing (e.g. {sorted(missing)[:3]})")
        if extra:
            r.errors.append(f"{name}: {len(extra)} unknown source1_entity_id rows (e.g. {sorted(extra)[:3]})")
    bad_prefix, dup, unknown = 0, 0, 0
    for ids in rows.values():
        dup += len(ids) - len(set(ids))
        bad_prefix += sum(not t.startswith(("S2-", "S3-")) for t in ids)
        if valid_other is not None:
            unknown += sum(t not in valid_other for t in ids)
    if dup:
        r.errors.append(f"{name}: {dup} duplicate ids inside ID lists")
    if bad_prefix:
        r.errors.append(f"{name}: {bad_prefix} ids are not S2-/S3- (self-matches to S1 are rejected)")
    if unknown:
        r.errors.append(f"{name}: {unknown} ids don't exist in test source2/source3")


def validate_outputs(matching: str | Path, candidate: str | Path | None = None, test_dir: str | Path | None = None,
                     s1_ids: Iterable[str] | None = None) -> Report:
    """Validate both output TSVs against every organiser rule, including matches being a subset of candidates."""
    r = Report()
    valid_s1 = set(s1_ids) if s1_ids is not None else None
    valid_other = None
    if test_dir is not None:
        s1, s2, s3 = load_sources(test_dir, usecols=["entity_id"])   # IDs only (low memory)
        valid_s1 = set(s1["entity_id"])
        valid_other = set(s2["entity_id"]) | set(s3["entity_id"])
    elif valid_s1 is None:
        r.warnings.append("no --test-dir: S1 coverage and id existence NOT checked")

    m = _parse_file(Path(matching), MATCH_HEADER, r)
    if m is not None:
        _check_lists("matching_results", m, valid_s1, valid_other, r)
        n = max(len(m), 1)
        sizes = [len(v) for v in m.values()]
        r.info.append(f"matching: {len(m)} rows | {sum(s == 0 for s in sizes) / n:.1%} empty | "
                      f"mean {sum(sizes) / n:.2f} ids/row | max {max(sizes, default=0)}")
    if candidate is not None:
        c = _parse_file(Path(candidate), CAND_HEADER, r)
        if c is not None:
            _check_lists("candidate_pairs", c, valid_s1, valid_other, r)
            n = max(len(c), 1)
            r.info.append(f"candidates: mean {sum(len(v) for v in c.values()) / n:.1f} ids/row")
            if m is not None:
                not_sub = sum(len(set(v) - set(c.get(k, ()))) for k, v in m.items())
                if not_sub:
                    r.errors.append(f"{not_sub} matched ids are not in candidate_pairs (pipeline bug)")
    return r


def main(argv=None) -> int:
    """CLI: validate a pair of output files."""
    ap = argparse.ArgumentParser(prog="python -m src.er_submission")
    sp = ap.add_subparsers(dest="cmd", required=True)
    v = sp.add_parser("validate")
    v.add_argument("--matching", required=True)
    v.add_argument("--candidate")
    v.add_argument("--test-dir", help="folder with test_source{1,2,3}.tsv (strongly recommended)")
    a = ap.parse_args(argv)
    rep = validate_outputs(a.matching, a.candidate, a.test_dir)
    print(rep)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
