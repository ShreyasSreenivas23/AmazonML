"""Write the shared validation folds (every model uses the same file; see docs/DECISIONS.md).

    python scripts/make_folds.py   # -> data/cache/folds_s1_k5.parquet [s1_id, fold, country, dev]

5 random folds over train S1 entities (seed 42). `dev` marks a fixed 100k-S1 sample of fold 0 for fast iteration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cv import make_folds  # noqa: E402
from src.er_data import cache_dir, load  # noqa: E402

SEED, K, DEV_N = 42, 5, 100_000


def main() -> None:
    s1, _, _, _ = load("train", with_gt=False)
    folds = make_folds(s1.rename(columns={"entity_id": "s1_id"}), n_splits=K, method="kfold", id_col="s1_id", seed=SEED)
    folds["country"] = s1.country.to_numpy()
    rng = np.random.default_rng(SEED)
    f0 = np.flatnonzero(folds.fold.to_numpy() == 0)
    folds["dev"] = False
    folds.loc[rng.choice(f0, size=min(DEV_N, len(f0)), replace=False), "dev"] = True
    out = cache_dir() / "folds_s1_k5.parquet"
    folds.to_parquet(out, index=False)
    print(folds.groupby("fold").agg(n=("s1_id", "size"), india=("country", lambda c: (c == "India").mean())))
    print(f"dev: {folds.dev.sum():,} S1 (fold 0) -> {out}")


if __name__ == "__main__":
    main()
