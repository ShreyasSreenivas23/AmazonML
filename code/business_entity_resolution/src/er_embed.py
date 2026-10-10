"""Multilingual sentence embeddings for blocking (multilingual-e5, MIT licence).

    python -m src.er_embed --split train --sources 2 3 --view name --model small     # resumable, heartbeat
    python -m src.er_embed --bench                                                   # throughput check

Views (EDA: names can be gibberish, so both matter):
    name  -> "query: <business_name>"
    addr  -> "query: <business_address>"
    both  -> "query: <business_name>, <business_address>"
Raw text is used (the model handles Devanagari/Kannada/French directly). Output: float16 L2-normalised matrix
aligned with the cached parquet row order: data/cache/emb/<split>_s<k>_<view>_<model>.npy
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.er_data import cache_dir

MODELS = {"small": "intfloat/multilingual-e5-small", "base": "intfloat/multilingual-e5-base"}
MAX_LEN = {"name": 48, "addr": 64, "both": 96}


def emb_dir() -> Path:
    """Folder for embedding caches (DATA_DIR/cache/emb)."""
    d = cache_dir() / "emb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def view_texts(df: pd.DataFrame, view: str) -> list[str]:
    """Texts for an embedding view: name, address, or name + address, with the e5 'query:' prefix."""
    if view == "name":
        return ("query: " + df.business_name).tolist()
    if view == "addr":
        return ("query: " + df.business_address).tolist()
    return ("query: " + df.business_name + ", " + df.business_address).tolist()


class Encoder:
    """Batched sentence encoder (multilingual e5 or a fine-tuned checkpoint) returning float16 embeddings."""
    def __init__(self, model: str = "small", device: str | None = None):
        """device=None: CUDA if available, else CPU with a warning. Embedding ~24M texts needs a GPU in practice
        (~21k texts/s on an RTX 5080; CPU is orders of magnitude slower)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                print("WARNING: no CUDA GPU found; embedding on CPU will be very slow", flush=True)
        repo = MODELS.get(model, model)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.dim = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str], max_len: int, batch: int = 1024) -> np.ndarray:
        """L2-normalised mean-pooled embeddings (float16), in input order. Length-sorted batching for speed."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), self.dim), dtype=np.float16)
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            x = self.tok([texts[j] for j in idx], max_length=max_len, truncation=True, padding=True,
                         return_tensors="pt").to(self.device)
            h = self.model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out[idx] = torch.nn.functional.normalize(e.float(), dim=-1).half().cpu().numpy()
        return out


def embed_file(split: str, k: int, view: str, model: str, chunk: int = 1_000_000, batch: int = 1024,
               tag: str | None = None) -> Path:
    """Embed one source file in resumable chunks, then concatenate. Writes a heartbeat for monitor.watch."""
    from monitor import Heartbeat

    df = pd.read_parquet(cache_dir() / f"{split}_s{k}.parquet", columns=["business_name", "business_address"])
    tag = tag or model   # filename suffix; use --tag for models loaded from a folder
    final = emb_dir() / f"{split}_s{k}_{view}_{tag}.npy"
    if final.exists():
        print(f"exists: {final.name}")
        return final
    parts_dir = emb_dir() / f".parts_{split}_s{k}_{view}_{tag}"
    parts_dir.mkdir(exist_ok=True)
    enc = Encoder(model)
    n_chunks = (len(df) + chunk - 1) // chunk
    run_id = os.environ.get("RUN_ID", f"embed-{split}-s{k}-{view}-{tag}")
    with Heartbeat(run_id, total_steps=n_chunks, every_steps=1, metric_name="rows/s") as hb:
        for c in range(n_chunks):
            part = parts_dir / f"{c:04d}.npy"
            if part.exists():
                continue
            t = time.time()
            sl = df.iloc[c * chunk:(c + 1) * chunk]
            np.save(part, enc.encode(view_texts(sl, view), MAX_LEN[view], batch))
            rate = len(sl) / (time.time() - t)
            hb.step(c + 1, samples=len(sl), rows_per_s=round(rate))
            print(f"  chunk {c + 1}/{n_chunks}: {rate:,.0f} rows/s", flush=True)
    mat = np.concatenate([np.load(parts_dir / f"{c:04d}.npy") for c in range(n_chunks)])
    assert len(mat) == len(df)
    np.save(final, mat)
    for p in parts_dir.glob("*.npy"):
        p.unlink()
    parts_dir.rmdir()
    print(f"wrote {final.name} {mat.shape}")
    return final


def bench(n: int = 100_000) -> None:
    """Measure encoding throughput on a sample (texts per second)."""
    df = pd.read_parquet(cache_dir() / "train_s2.parquet").sample(n, random_state=0)
    for model in ("small", "base"):
        enc = Encoder(model)
        for view in ("name", "addr", "both"):
            texts = view_texts(df, view)
            enc.encode(texts[:2048], MAX_LEN[view])  # warm-up
            torch.cuda.synchronize()
            t = time.time()
            enc.encode(texts, MAX_LEN[view])
            torch.cuda.synchronize()
            dt = time.time() - t
            print(f"{model:5} {view:4}: {n / dt:8,.0f} rows/s -> 10.3M train S2+S3 in {10.3e6 / (n / dt) / 60:5.1f} min "
                  f"| peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del enc
        torch.cuda.empty_cache()


def main() -> None:
    """Embed the requested sources/views of a split and write resumable per-chunk caches."""
    ap = argparse.ArgumentParser(prog="python -m src.er_embed")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sources", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--view", default="name", choices=list(MAX_LEN))
    ap.add_argument("--model", default="small", help="small | base | path to a fine-tuned model folder")
    ap.add_argument("--tag", default=None, help="output filename suffix (default: the model key)")
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    if a.bench:
        bench()
        return
    for k in a.sources:
        embed_file(a.split, k, a.view, a.model, batch=a.batch, tag=a.tag)


if __name__ == "__main__":
    main()
