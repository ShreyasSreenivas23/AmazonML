"""Character n-gram "view" for blocking: char TF-IDF of the transliterated core name, reduced by SVD to 256-d.

Why (recall analysis, 2026-09-25): 47% of true pairs missed by the e5 views have an EMPTY candidate address, and
most of the rest are Indic-script names whose transliteration shares character fragments with the S1 name
("sevn impeks" vs "seven impex"). Semantic embeddings miss these; character n-grams catch them.

    python -m src.er_charvec --splits train test      # writes data/cache/emb/<split>_s<k>_char_small.npy

Output files use the embedding naming scheme (view "char", suffix "small") so the existing exact GPU kNN and
Split(views=...) machinery can use them unchanged. L2-normalised float16, row order = cached parquet order.
Fitted on train text only (label-free), then applied to test.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from src.er_data import cache_dir
from src.er_embed import emb_dir

DIM = 256
HASH = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**18, alternate_sign=False,
                         norm=None, lowercase=False)


def texts(split: str, k: int) -> list[str]:
    """Transliterated core name (already ASCII-normalised in the cache)."""
    return pd.read_parquet(cache_dir() / f"{split}_s{k}_norm.parquet", columns=["name_core"]).name_core.tolist()


def fit(sample_n: int = 1_000_000, seed: int = 0) -> tuple[TfidfTransformer, TruncatedSVD]:
    """IDF + SVD fitted on a sample of train names from all three sources (text only, no labels)."""
    rng = np.random.default_rng(seed)
    pool = []
    for k in (1, 2, 3):
        t = texts("train", k)
        pool += [t[i] for i in rng.choice(len(t), size=min(sample_n // 3, len(t)), replace=False)]
    X = HASH.transform(pool)
    tfidf = TfidfTransformer(sublinear_tf=True).fit(X)
    svd = TruncatedSVD(DIM, algorithm="randomized", n_iter=4, random_state=seed).fit(tfidf.transform(X))
    return tfidf, svd


def transform(tfidf: TfidfTransformer, svd: TruncatedSVD, t: list[str], chunk: int = 500_000) -> np.ndarray:
    """Char n-gram TF-IDF vectors (L2-normalised, float16) for a list of strings."""
    out = np.empty((len(t), DIM), np.float16)
    for s in range(0, len(t), chunk):
        z = svd.transform(tfidf.transform(HASH.transform(t[s:s + chunk])))
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
        out[s:s + chunk] = z.astype(np.float16)
    return out


def main() -> None:
    """Write char n-gram embedding caches for the requested splits."""
    ap = argparse.ArgumentParser(prog="python -m src.er_charvec")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    t0 = time.time()
    tfidf, svd = fit()
    print(f"fitted char tf-idf + svd({DIM}) [{time.time() - t0:.0f}s], explained var {svd.explained_variance_ratio_.sum():.3f}")
    for split in a.splits:
        for k in (1, 2, 3):
            v = transform(tfidf, svd, texts(split, k))
            np.save(emb_dir() / f"{split}_s{k}_char_small.npy", v)
            print(f"{split} S{k}: {v.shape} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
