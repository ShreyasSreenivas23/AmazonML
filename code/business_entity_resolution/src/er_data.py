"""Fast, exact loading of the organiser TSVs + parquet cache.

    python -m src.er_data            # build data/cache/*.parquet once (~1 min), verifies row counts
    from src.er_data import load     # s1, s2, s3, gt_pairs = load("train")

Reads with pyarrow (multithreaded), tab delimiter, **no quote handling** (organiser files are unquoted), every
column as string, empty stays "" (never NaN). Ground truth is also exploded to one row per (s1_id, cand_id) pair.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.config import get_paths

COLS = ["entity_id", "business_name", "business_address", "country"]


def dataset_dir() -> Path:
    """Folder holding the organisers' dataset (DATA_DIR/dataset)."""
    return get_paths().data / "dataset"


def cache_dir() -> Path:
    """Folder for parquet caches derived from the raw TSVs (created if missing)."""
    d = get_paths().data / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_tsv_arrow(path: Path) -> pa.Table:
    """Read one TSV exactly as strings with pyarrow (tab separator, no quoting, no NA parsing)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                             strings_can_be_null=False, quoted_strings_can_be_null=False),
        read_options=pacsv.ReadOptions(use_threads=True, block_size=64 << 20),
    )


def count_lines(path: Path) -> int:
    """Count data lines in a file (used to verify every row was read)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def build_cache(split: str, verify: bool = True) -> None:
    """Convert a split's TSVs to parquet (and explode ground truth into pairs), checking row counts."""
    src, out = dataset_dir() / split, cache_dir()
    for k in (1, 2, 3):
        path = src / f"{split}_source{k}.tsv"
        t = read_tsv_arrow(path)
        assert t.column_names == COLS, (path, t.column_names)
        if verify:
            assert t.num_rows == count_lines(path) - 1, f"row count mismatch in {path}"
        pq.write_table(t, out / f"{split}_s{k}.parquet")
        print(f"  {path.name}: {t.num_rows:,} rows")
    gt_path = src / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = read_tsv_arrow(gt_path)
        if verify:
            assert gt.num_rows == count_lines(gt_path) - 1
        pq.write_table(gt, out / f"{split}_gt.parquet")
        lists = pc.split_pattern(gt["matched_entity_ids"], ",")
        pairs = pa.table({"s1_id": pc.list_parent_indices(lists), "cand_id": pc.list_flatten(lists)})
        pairs = pairs.filter(pc.not_equal(pairs["cand_id"], ""))
        pairs = pairs.set_column(0, "s1_id", pc.take(gt["source1_entity_id"], pairs["s1_id"]))
        pq.write_table(pairs, out / f"{split}_gt_pairs.parquet")
        print(f"  {gt_path.name}: {gt.num_rows:,} S1 rows, {pairs.num_rows:,} match pairs")


def load(split: str, with_gt: bool = True):
    """Return (s1, s2, s3, gt_pairs|None) as pandas DataFrames of str (build the cache first)."""
    d = cache_dir()
    frames = [pd.read_parquet(d / f"{split}_s{k}.parquet") for k in (1, 2, 3)]
    gt = None
    if with_gt and (d / f"{split}_gt_pairs.parquet").exists():
        gt = pd.read_parquet(d / f"{split}_gt_pairs.parquet")
    return (*frames, gt)


def main() -> None:
    """Build the parquet caches for train and test."""
    ap = argparse.ArgumentParser(prog="python -m src.er_data")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    for s in a.splits:
        t = time.time()
        print(f"[{s}]")
        build_cache(s, verify=not a.no_verify)
        print(f"  done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
