"""Exact GPU/CPU top-k and country-partitioned dense candidates."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from src.er_blocking import blocking_recall, dense_candidates, gpu_topk  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def test_gpu_topk_matches_bruteforce_across_chunks():
    rng = np.random.default_rng(0)
    q, keys = _unit(rng.normal(size=(50, 16))), _unit(rng.normal(size=(1000, 16)))
    ii, ss = gpu_topk(q, keys, k=7, q_batch=16, key_chunk=128, device=DEV)   # forces multi-chunk merging
    ref = q.astype(np.float32) @ keys.astype(np.float32).T
    ref_i = np.argsort(-ref, axis=1)[:, :7]
    # compare as sets of scores (fp16 ties can reorder indices)
    np.testing.assert_allclose(np.sort(ss, 1), np.sort(np.take_along_axis(ref, ref_i, 1), 1), atol=2e-2)
    assert (ss[:, :-1] >= ss[:, 1:] - 1e-6).all()


def test_dense_candidates_stay_within_country_and_recall():
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=(6, 8)))
    queries = pd.DataFrame({"entity_id": [f"S1-{i}" for i in range(6)], "country": ["US"] * 3 + ["France"] * 3})
    pool = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(6)], "country": ["US"] * 3 + ["France"] * 3})
    noisy = _unit(base.astype(np.float32) + 0.01 * rng.normal(size=base.shape))
    c = dense_candidates(queries, pool, base, noisy, k=2, view="name")
    ctry = dict(zip(pool.entity_id, pool.country))
    q_ctry = dict(zip(queries.entity_id, queries.country))
    assert all(q_ctry[a] == ctry[b] for a, b in zip(c.s1_id, c.cand_id))
    gt = pd.DataFrame({"s1_id": queries.entity_id, "cand_id": pool.entity_id})
    r = blocking_recall(c, gt, queries.entity_id, ks=(1, 2))
    assert r["recall@1"] == 1.0 and r["cands@2"] == 2.0
