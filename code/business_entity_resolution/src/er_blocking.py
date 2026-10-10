"""Candidate generation (blocking). Contract: docs/WORKERS.md.

Dense retriever: exact top-K cosine search on the GPU (chunked matmul, no approximation), partitioned by the
`country` string (EDA: 0 cross-country matches in 7.64M train pairs; generic, so France needs no special case).

    idx, score = gpu_topk(queries_f16, keys_f16, k=50)
    cands = dense_candidates(s1, others, q_emb, k_emb, k=50, view="name")   # DataFrame[s1_id, cand_id, view, rank, score]
    blocking_recall(cands, gt_pairs, s1_ids, ks=[1, 5, 10, 20, 50])
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.inference_mode()
def gpu_topk(q: np.ndarray, keys: np.ndarray, k: int, q_batch: int = 2048, key_chunk: int = 1_000_000,
             device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by dot product (= cosine for L2-normalised rows). Returns (indices int64, scores float32)."""
    k = min(k, len(keys))
    if len(q) == 0 or k == 0:
        return np.empty((len(q), 0), np.int64), np.empty((len(q), 0), np.float32)
    kt = [torch.from_numpy(keys[i:i + key_chunk]).to(device, torch.float16) for i in range(0, len(keys), key_chunk)]
    offs = np.cumsum([0] + [len(c) for c in kt])[:-1]
    out_i = np.empty((len(q), k), np.int64)
    out_s = np.empty((len(q), k), np.float32)
    for i in range(0, len(q), q_batch):
        qb = torch.from_numpy(q[i:i + q_batch]).to(device, torch.float16)
        best_s, best_i = None, None
        for off, kc in zip(offs, kt):
            s = qb @ kc.T                                   # (b, chunk) fp16
            kk = min(k, s.shape[1])
            cs, ci = s.topk(kk, dim=1)
            ci = ci + int(off)
            if best_s is None:
                best_s, best_i = cs.float(), ci
            else:
                cat_s, cat_i = torch.cat([best_s, cs.float()], 1), torch.cat([best_i, ci], 1)
                best_s, sel = cat_s.topk(k, dim=1)
                best_i = cat_i.gather(1, sel)
        out_i[i:i + len(qb)] = best_i.cpu().numpy()
        out_s[i:i + len(qb)] = best_s.cpu().numpy()
    del kt
    torch.cuda.empty_cache()
    return out_i, out_s


def dense_candidates(queries: pd.DataFrame, pool: pd.DataFrame, q_emb: np.ndarray, p_emb: np.ndarray, k: int,
                     view: str, by: str = "country") -> pd.DataFrame:
    """Top-k pool records per query row, searched only within the same `by` value.

    queries/pool: frames with entity_id and `by`; q_emb/p_emb: embeddings aligned with those frames' rows.
    Returns DataFrame[s1_id, cand_id, view, rank (0-based), score].
    """
    frames = []
    for val, qg in queries.groupby(by, sort=False):
        pmask = (pool[by] == val).to_numpy()
        if not pmask.any():
            continue
        pidx = np.flatnonzero(pmask)
        qpos = queries.index.get_indexer(qg.index)
        ii, ss = gpu_topk(q_emb[qpos], p_emb[pidx], k)
        kk = ii.shape[1]
        frames.append(pd.DataFrame({
            "s1_id": np.repeat(qg.entity_id.to_numpy(), kk),
            "cand_id": pool.entity_id.to_numpy()[pidx[ii.ravel()]],
            "view": view,
            "rank": np.tile(np.arange(kk, dtype=np.int16), len(qg)),
            "score": ss.ravel(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["s1_id", "cand_id", "view", "rank", "score"])


def blocking_recall(cands: pd.DataFrame, gt_pairs: pd.DataFrame, s1_ids, ks=(1, 5, 10, 20, 50)) -> dict:
    """Pair recall@k (true pairs of the given S1s whose candidate rank < k) and candidates/S1 at each k."""
    s1_ids = pd.Index(s1_ids)
    gt = gt_pairs[gt_pairs.s1_id.isin(s1_ids)]
    best = cands.groupby(["s1_id", "cand_id"], sort=False)["rank"].min().rename("best_rank").reset_index()
    m = gt.merge(best, on=["s1_id", "cand_id"], how="left")
    out = {"n_true_pairs": len(gt)}
    for k in ks:
        out[f"recall@{k}"] = float((m.best_rank < k).mean()) if len(gt) else float("nan")
        out[f"cands@{k}"] = float((best.best_rank < k).sum() / max(len(s1_ids), 1))
    return out
