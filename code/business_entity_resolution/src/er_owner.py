"""Listwise owner model (OW04): which S1 owns a contested pool record?

A pool record c (S2/S3) is contested when several S1s retrieved it. Pair cross-encoders score (S1, c) in isolation,
but the dominant remaining error (E015 analysis: ~85% of dev loss = true matches rejected because near-identical
records of OTHER S1s compete) needs the competitors in view. This model reads c together with its top-K competing
S1s in ONE sequence, so attention can compare competitors directly (house number, legal form, word order), and
predicts a softmax over the K slots + "none" (owner absent from the list, or c is a distractor).

Sequence (XLM-R style specials): <s> record </s> </s> S1_1 </s> </s> S1_2 ... </s>. Each segment is mean-pooled;
slot logit = MLP([seg; record; seg*record]), none logit = MLP(<s>). Slots are shuffled in training, so the model has
to use the text, not the stage-1 rank order.

Leakage protocol: training records are only those that NO fold-0 S1 retrieved and whose true owner is not in fold 0.
Every fold-0 pair (stage-2 fit pool + dev) is therefore scored out-of-sample, like test.

    python scripts/owner_pairs.py --out runs/OW04          # -> pairs / fold0 / test parquet from the stage-1 chunks
    python -m src.er_owner prep  --pairs runs/OW04/pairs.parquet --frame runs/OW04/fold0.parquet         --test-frame runs/OW04/test.parquet --out runs/OW04
    python -m src.er_owner train --dir runs/OW04 --model intfloat/multilingual-e5-large --epochs 2 --batch 32         --lr 2e-5 --n 200000 --bf16
    python -m src.er_owner score --dir runs/OW04 --split train --batch 256 --bf16   # -> runs/OW04/train_owner.parquet
    python -m src.er_owner score --dir runs/OW04 --split test  --batch 256 --bf16   # -> runs/OW04/test_owner.parquet
    python scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_p      --out runs/OW04p
    python scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_margin --out runs/OW04m
Stage 2 consumes runs/OW04p and runs/OW04m as two more --ce-dir feature directories (E034).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "intfloat/multilingual-e5-small"     # MIT, 118M; small enough for CPU (bf16/AMX) training
K_MAX = 6


# ------------------------------------------------------------------------------------------------------ groups
def build_groups(pairs: pd.DataFrame, k_max: int = K_MAX, min_prob: float = 0.01, contest_min: float = 0.05,
                 records: set | None = None) -> pd.DataFrame:
    """Contested records and their competing S1s.

    pairs: [s1_id, cand_id, prob] over the WHOLE split (every S1 that retrieved each record). A record's list = its S1s
    with prob >= min_prob, sorted by prob desc, top k_max. Kept when the 2nd-best prob >= contest_min (a real contest).
    records: optional subset of cand_ids to keep. Returns [cand_id, s1_ids (list), probs (list)].
    """
    p = pairs[["s1_id", "cand_id", "prob"]]
    p = p[p.prob.to_numpy() >= min_prob]
    if records is not None:
        p = p[p.cand_id.isin(records)]
    p = p.sort_values(["cand_id", "prob"], ascending=[True, False], kind="stable")
    rank = p.groupby("cand_id", sort=False).cumcount().to_numpy()
    p = p[rank < k_max]
    second = p.groupby("cand_id", sort=False).prob.nth(1)
    ok = set(p.loc[second.index[second.to_numpy() >= contest_min], "cand_id"]) if len(second) else set()
    p = p[p.cand_id.isin(ok)]
    g = p.groupby("cand_id", sort=False).agg(s1_ids=("s1_id", list), probs=("prob", list)).reset_index()
    return g


def label_groups(g: pd.DataFrame, owner: pd.Series) -> np.ndarray:
    """Slot index of each record's true owner in its list, or len(list)... mapped to K_MAX (the 'none' slot)."""
    own = owner.reindex(g.cand_id).to_numpy()
    lab = np.full(len(g), K_MAX, np.int64)
    for i, (lst, o) in enumerate(zip(g.s1_ids, own)):
        if isinstance(o, str) and o in lst:
            lab[i] = lst.index(o)
    return lab


def exclude_fold0(g: pd.DataFrame, owner: pd.Series, fold: pd.Series) -> pd.DataFrame:
    """Training groups: drop every group whose SLOT LIST contains a fold-0 S1, or whose true owner is in fold 0.

    A fold-0 pair only ever receives an owner score when its S1 is in the record's slot list, so this keeps every
    fold-0 (stage-2 fit + dev) owner score out-of-sample, without discarding the crowded records a stricter rule
    (any fold-0 S1 in the full top-15 retrieval) removed: OW01 kept only 48k groups that way."""
    f0 = set(fold.index[fold.to_numpy() == 0])
    own = owner.reindex(g.cand_id).to_numpy()
    keep = np.array([not (set(lst) & f0) and not (isinstance(o, str) and o in f0) for lst, o in zip(g.s1_ids, own)],
                    dtype=bool)
    return g[keep].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ sequences
def make_example(rec: list[int], slots: list[list[int]], bos: int, eos: int) -> tuple[list[int], list[int]]:
    """(input_ids, seg) for one record + its slot token lists. seg: 0 special, 1 record, 2+j slot j."""
    ids, seg = [bos] + rec + [eos], [0] + [1] * len(rec) + [0]
    for j, s in enumerate(slots):
        ids += [eos] + s + [eos]
        seg += [0] + [2 + j] * len(s) + [0]
    return ids, seg


def collate(examples: list[tuple[list[int], list[int]]], pad: int):
    """Pad to the batch max length -> (input_ids, attention_mask, seg) LongTensors."""
    import torch

    L = max(len(e[0]) for e in examples)
    ids = torch.full((len(examples), L), pad, dtype=torch.long)
    seg = torch.zeros((len(examples), L), dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (x, s) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x)
        seg[i, :len(s)] = torch.tensor(s)
        att[i, :len(x)] = 1
    return ids, att, seg


def owner_model(encoder, hidden: int):
    """Encoder + slot/none heads. forward(ids, att, seg, n_slots) -> logits [B, K_MAX + 1] (last = none)."""
    import torch
    from torch import nn

    class OwnerModel(nn.Module):
        """Listwise owner scorer on top of a transformer encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.enc = encoder
            self.slot = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.none = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        def forward(self, ids, att, seg, n_slots):
            """Mean-pool each segment, score slots against the record, mask empty slots."""
            h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state.float()
            oh = nn.functional.one_hot(seg, K_MAX + 2).float()                  # B,T,K+2
            means = torch.einsum("btk,bth->bkh", oh, h) / oh.sum(1).clamp(min=1.0)[..., None]
            rec, slots = means[:, 1], means[:, 2:]
            r = rec[:, None].expand_as(slots)
            z = self.slot(torch.cat([slots, r, slots * r], -1)).squeeze(-1)    # B,K
            valid = torch.arange(K_MAX, device=ids.device)[None] < n_slots[:, None]
            z = z.masked_fill(~valid, -1e4)
            return torch.cat([z, self.none(h[:, 0])], 1)

    return OwnerModel()


# ----------------------------------------------------------------------------------------------------- texts
def token_table(ids: np.ndarray, left: pd.DataFrame, right: pd.DataFrame, tok, seg_len: int) -> dict[str, list[int]]:
    """entity_id -> token ids of 'name | address' (no specials, truncated), for every id needed."""
    from src.er_crossenc import record_text

    ids = pd.unique(ids)
    src = pd.concat([left.reindex(ids[pd.Series(ids).str.startswith("S1-").to_numpy()]),
                     right.reindex(ids[~pd.Series(ids).str.startswith("S1-").to_numpy()])])
    texts = [record_text(n, a) for n, a in zip(src.business_name.fillna(""), src.business_address.fillna(""))]
    out: dict[str, list[int]] = {}
    for i in range(0, len(texts), 50_000):
        enc = tok(texts[i:i + 50_000], add_special_tokens=False, truncation=True, max_length=seg_len)["input_ids"]
        out.update(zip(src.index[i:i + 50_000], enc))
    return out


# ------------------------------------------------------------------------------------------------------- CLI
def cmd_prep(a: argparse.Namespace) -> None:
    """Build training groups (labelled) and inference groups for the fold-0 frame and the test frame."""
    from src.er_data import cache_dir

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs = pd.read_parquet(a.pairs, columns=["s1_id", "cand_id", "prob"])
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id").fold
    gt = pd.read_parquet(cache_dir() / "train_gt_pairs.parquet")
    owner = gt.drop_duplicates("cand_id").set_index("cand_id").s1_id
    allg = build_groups(pairs, a.k_max, a.min_prob, a.contest_min)
    tr = exclude_fold0(allg, owner, folds)
    tr["label"] = label_groups(tr, owner)
    tr.to_parquet(out / "train_groups.parquet", index=False)
    print(f"train groups {len(tr):,} of {len(allg):,} contested; none-share {(tr.label == K_MAX).mean():.3f};"
          f" slots/record {tr.s1_ids.str.len().mean():.2f} [{time.time() - t0:.0f}s]", flush=True)
    del allg
    frame_recs = set(pd.read_parquet(a.frame, columns=["cand_id"]).cand_id)
    ig = build_groups(pairs, a.k_max, a.min_prob, a.contest_min, records=frame_recs)
    ig.to_parquet(out / "infer_train_groups.parquet", index=False)
    print(f"fold-0 frame inference groups {len(ig):,} [{time.time() - t0:.0f}s]", flush=True)
    if a.test_frame and Path(a.test_frame).exists():
        tp = pd.read_parquet(a.test_frame, columns=["s1_id", "cand_id", "prob"])
        tg = build_groups(tp, a.k_max, a.min_prob, a.contest_min)
        tg.to_parquet(out / "infer_test_groups.parquet", index=False)
        print(f"test inference groups {len(tg):,} [{time.time() - t0:.0f}s]", flush=True)
    (out / "prep.json").write_text(json.dumps(vars(a), indent=2))


def _setup(threads: int, model_name: str = MODEL):
    """Torch thread count + model/tokenizer for CPU (or GPU if present)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch, tok, enc, dev


def cmd_train(a: argparse.Namespace) -> None:
    """Train on up to --n labelled groups (1 epoch, shuffled slots), checkpointing every --ckpt-every steps."""
    from src.er_crossenc import load_sources

    d = Path(a.dir)
    torch, tok, enc, dev = _setup(a.threads, a.model)
    g = pd.read_parquet(d / "train_groups.parquet")
    if len(g) > a.n:
        g = g.sample(a.n, random_state=0).reset_index(drop=True)
    left, right = load_sources("train")
    table = token_table(np.concatenate([g.cand_id.to_numpy(), np.concatenate(g.s1_ids.to_numpy())]),
                        left, right, tok, a.seg_len)
    del left, right
    model = owner_model(enc, enc.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(g) / a.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    ck = d / "ckpt.pt"
    step = 0
    if ck.exists():
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed at step {step}/{steps}", flush=True)
    glen = np.array([len(table[c]) + sum(len(table[x]) + 2 for x in lst) for c, lst in zip(g.cand_id, g.s1_ids)])
    mega = 50 * a.batch
    batches = []
    for ep in range(a.epochs):                                        # length-bucketed, random batch order per epoch
        rng = np.random.default_rng(ep)
        perm = rng.permutation(len(g))
        perm = np.concatenate([m[np.argsort(glen[m], kind="stable")]
                               for m in np.array_split(perm, max(1, len(g) // mega))])
        eb = [perm[i:i + a.batch] for i in range(0, len(perm), a.batch)]
        batches += [eb[j] for j in rng.permutation(len(eb))]
    steps = len(batches)
    bos, eos, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    t0, run_loss, run_acc, seen = time.time(), 0.0, 0.0, 0
    while step < steps:
        ix = batches[step]
        exs, labs, ns = [], [], []
        for i in ix:
            lst, lab = list(g.s1_ids[i]), int(g.label[i])
            sp = np.random.default_rng(step * 100_003 + int(i)).permutation(len(lst))       # shuffled slots
            lst = [lst[p] for p in sp]
            lab = K_MAX if lab == K_MAX else int(np.flatnonzero(sp == lab)[0])
            exs.append(make_example(table[g.cand_id[i]], [table[s] for s in lst], bos, eos))
            labs.append(lab)
            ns.append(len(lst))
        ids, att, seg = (x.to(dev) for x in collate(exs, pad))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.bf16):
            logits = model(ids, att, seg, torch.tensor(ns, device=dev))
        y = torch.tensor(labs, device=dev)
        loss = loss_fn(logits.float(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run_loss += loss.item() * len(ix)
        run_acc += (logits.argmax(1) == y).float().sum().item()
        seen += len(ix)
        if step % 50 == 0 or step == steps:
            el = time.time() - t0
            print(f"step {step}/{steps} loss {run_loss / seen:.4f} acc {run_acc / seen:.4f} "
                  f"{seen / el:.1f} groups/s eta {(steps - step) * a.batch / max(seen / el, 1e-9) / 60:.0f} min", flush=True)
            run_loss = run_acc = 0.0
            seen, t0 = 0, time.time()
        if step % a.ckpt_every == 0 or step == steps:
            tmp = ck.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, tmp)
            os.replace(tmp, ck)
    torch.save(model.state_dict(), d / "owner_model.pt")
    (d / "train.json").write_text(json.dumps({"groups": len(g), "steps": steps, "device": dev.type, **vars(a)}, indent=2))
    print("TRAIN DONE", flush=True)


def cmd_score(a: argparse.Namespace) -> None:
    """Owner features for every slot of every inference group -> <split>_owner.parquet [s1_id, cand_id, own_*]."""
    from src.er_crossenc import load_sources

    d = Path(a.dir)
    tj = d / "train.json"
    model_name = json.loads(tj.read_text()).get("model", MODEL) if tj.exists() else MODEL
    torch, tok, enc, dev = _setup(a.threads, model_name)
    gp = d / f"infer_{a.split}_groups.parquet"
    if not gp.exists() and a.split == "test" and a.test_frame:
        prep = json.loads((d / "prep.json").read_text())
        tp = pd.read_parquet(a.test_frame, columns=["s1_id", "cand_id", "prob"])
        build_groups(tp, prep["k_max"], prep["min_prob"], prep["contest_min"]).to_parquet(gp, index=False)
        del tp
    g = pd.read_parquet(gp)
    print(f"{a.split}: {len(g):,} contested groups", flush=True)
    left, right = load_sources(a.split)
    table = token_table(np.concatenate([g.cand_id.to_numpy(), np.concatenate(g.s1_ids.to_numpy())]),
                        left, right, tok, a.seg_len)
    del left, right
    model = owner_model(enc, enc.config.hidden_size).to(dev)
    model.load_state_dict(torch.load(d / "owner_model.pt", map_location=dev))
    model.eval()
    bos, eos, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    lens = np.array([len(table[c]) + sum(len(table[s]) for s in lst) for c, lst in zip(g.cand_id, g.s1_ids)])
    order = np.argsort(lens, kind="stable")                        # length-sorted batches -> little padding
    probs = np.zeros((len(g), K_MAX + 1), np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for b, st in enumerate(range(0, len(order), a.batch)):
            ix = order[st:st + a.batch]
            exs = [make_example(table[g.cand_id[i]], [table[s] for s in g.s1_ids[i]], bos, eos) for i in ix]
            ids, att, seg = (x.to(dev) for x in collate(exs, pad))
            ns = torch.tensor([len(g.s1_ids[i]) for i in ix], device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.bf16):
                logits = model(ids, att, seg, ns)
            probs[ix] = torch.softmax(logits.float(), 1).cpu().numpy()
            if b % 200 == 0:
                done = st + len(ix)
                print(f"{done:,}/{len(g):,} groups {done / (time.time() - t0):.1f}/s", flush=True)
    rows = []
    for i, (c, lst) in enumerate(zip(g.cand_id, g.s1_ids)):
        p = probs[i, :len(lst)]
        for j, s in enumerate(lst):
            other = np.delete(p, j)
            rows.append((s, c, p[j], probs[i, K_MAX], p[j] - (other.max() if len(other) else 0.0), len(lst)))
    out = pd.DataFrame(rows, columns=["s1_id", "cand_id", "own_p", "own_none", "own_margin", "own_n"])
    out.to_parquet(d / f"{a.split}_owner.parquet", index=False)
    print(f"SCORE DONE {a.split}: {len(out):,} pairs [{time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    """CLI: prep | train | score."""
    ap = argparse.ArgumentParser(prog="python -m src.er_owner")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--pairs", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--test-frame", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--k-max", type=int, default=K_MAX)
    p.add_argument("--min-prob", type=float, default=0.01)
    p.add_argument("--contest-min", type=float, default=0.05)
    for name in ("train", "score"):
        q = sp.add_parser(name)
        q.add_argument("--dir", required=True)
        q.add_argument("--seg-len", type=int, default=40)
        q.add_argument("--threads", type=int, default=0)
        q.add_argument("--bf16", action="store_true")
    t = sp.choices["train"]
    t.add_argument("--n", type=int, default=200_000)
    t.add_argument("--model", default=MODEL, help="HF encoder, e.g. intfloat/multilingual-e5-base (MIT) on a GPU")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--ckpt-every", type=int, default=200)
    s = sp.choices["score"]
    s.add_argument("--split", choices=["train", "test"], required=True)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--test-frame", default="", help="build infer_test_groups from this frame if missing")
    a = ap.parse_args()
    {"prep": cmd_prep, "train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
