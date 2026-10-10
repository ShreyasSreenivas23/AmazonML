"""Fine-tune multilingual-e5-small as a bi-encoder retriever on labelled train matches (in-batch contrastive).

Error analysis (E010 dev): 51% of remaining loss is true pairs never retrieved, 3x worse in India (Indic-script
names, truncated or empty addresses). Off-the-shelf e5 doesn't place those near their S1; training on our 7.6M
labelled pairs teaches exactly those equivalences.

    python -m src.er_biencoder train --out runs/E014-bienc/model --n 1500000       # folds 1-4 only (dev unseen)
    python -m src.er_embed --split train --sources 1 2 3 --view both --model runs/E014-bienc/model --tag ft

Loss: symmetric InfoNCE with in-batch negatives (scale 20). Batches are drawn from ONE country at a time so
negatives are realistic. Text = "query: <name>, <address>" on both sides (the e5 'both' view). Checkpoints every
`ckpt_every` steps (atomic) and resumes; heartbeat for monitor.watch.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.er_data import cache_dir, load

BASE = "intfloat/multilingual-e5-small"


def text(df: pd.DataFrame) -> pd.Series:
    """Model input for a record: 'query: <name>, <address>' (the e5 'both' view)."""
    return "query: " + df.business_name + ", " + df.business_address


def build_pairs(n: int, seed: int = 0) -> pd.DataFrame:
    """Sample (s1_text, match_text, country) from train GT pairs of folds 1-4 (never fold 0 = dev)."""
    s1, s2, s3, gt = load("train")
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").fold
    gt = gt[folds.reindex(gt.s1_id).to_numpy() != 0]
    gt = gt.sample(min(n, len(gt)), random_state=seed)
    L = s1.set_index("entity_id")
    R = pd.concat([s2, s3]).set_index("entity_id")
    return pd.DataFrame({"a": text(L.loc[gt.s1_id]).to_numpy(), "b": text(R.loc[gt.cand_id]).to_numpy(),
                         "country": L.country.reindex(gt.s1_id).to_numpy()})


def batches(pairs: pd.DataFrame, bs: int, seed: int) -> list[np.ndarray]:
    """Index batches, each from a single country, shuffled."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in pairs.groupby("country"):
        idx = rng.permutation(g.index.to_numpy())
        out += [idx[i:i + bs] for i in range(0, len(idx) - bs + 1, bs)]
    rng.shuffle(out)
    return out


def encode(model, tok, texts: list[str], max_len: int, device: str) -> torch.Tensor:
    """Mean-pooled, L2-normalised embeddings for a batch of texts."""
    x = tok(texts, max_length=max_len, truncation=True, padding=True, return_tensors="pt").to(device)
    h = model(**x).last_hidden_state
    m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)


def cmd_train(a: argparse.Namespace) -> None:
    """Train the bi-encoder with in-batch InfoNCE on folds 1-4 GT pairs (resumable, checkpointed)."""
    from monitor import Heartbeat

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.parquet"
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
    else:
        pairs = build_pairs(a.n, a.seed)
        pairs.to_parquet(pairs_path, index=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE).to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    bl = batches(pairs, a.batch, a.seed)
    total = len(bl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.06)
    step = 0
    ck = out / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step = s["step"]
        print(f"resumed at step {step}/{total}")
    model.train()
    t0 = time.time()
    with Heartbeat(os.environ.get("RUN_ID", "bienc-train"), total_steps=total, every_steps=50) as hb:
        for i in range(step, total):
            b = pairs.loc[bl[i]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                ea = encode(model, tok, b.a.tolist(), a.max_len, device)
                eb = encode(model, tok, b.b.tolist(), a.max_len, device)
                logits = (ea @ eb.T).float() * a.scale
            lab = torch.arange(len(b), device=device)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            hb.step(i + 1, loss=loss.item(), lr=sched.get_last_lr()[0], samples=len(b))
            if (i + 1) % a.ckpt_every == 0 or i + 1 == total:
                tmp = ck.with_suffix(".tmp")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": i + 1}, tmp)
                os.replace(tmp, ck)
            if (i + 1) % 200 == 0:
                print(f"step {i + 1}/{total} loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved bi-encoder to {out}")


def main() -> None:
    """CLI entry point for bi-encoder training."""
    ap = argparse.ArgumentParser(prog="python -m src.er_biencoder")
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=1_500_000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--scale", type=float, default=20.0)
    t.add_argument("--max-len", type=int, default=96)
    t.add_argument("--ckpt-every", type=int, default=500)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cmd_train(a)


if __name__ == "__main__":
    main()
