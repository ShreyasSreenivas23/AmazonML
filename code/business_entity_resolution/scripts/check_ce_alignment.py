"""Pre-flight for an external --ce-dir: every <split>_ce/NNN.parquet must be row-aligned with runs/E015/<split>_chunks
(same length, identical s1_id and cand_id order), exactly what er_fullpass.with_ce() requires. Prints non-NaN share.

    python scripts/check_ce_alignment.py runs/OW04p runs/OW04m
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ok = True
for ce_dir in sys.argv[1:]:
    for split in ("train", "test"):
        chunks = sorted(Path(f"runs/E015/{split}_chunks").glob("*.parquet"))
        files = sorted((Path(ce_dir) / f"{split}_ce").glob("*.parquet"))
        if len(files) != len(chunks):
            print(f"FAIL {ce_dir} {split}: {len(files)} files vs {len(chunks)} chunks"); ok = False; continue
        n = scored = 0
        for c, f in zip(chunks, files):
            a = pd.read_parquet(c, columns=["s1_id", "cand_id"])
            b = pd.read_parquet(f, columns=["s1_id", "cand_id", "ce_score"])
            if len(a) != len(b) or not (a.s1_id.to_numpy() == b.s1_id.to_numpy()).all() \
                    or not (a.cand_id.to_numpy() == b.cand_id.to_numpy()).all():
                print(f"FAIL {ce_dir} {split} {f.name}: rows misaligned with {c.name}"); ok = False
            n += len(b); scored += int(b.ce_score.notna().sum())
        print(f"{ce_dir} {split}: {len(files)} files, {n:,} rows, non-NaN {scored / max(n, 1):.4f}")
print("ALIGNMENT OK" if ok else "ALIGNMENT FAILED")
sys.exit(0 if ok else 1)
