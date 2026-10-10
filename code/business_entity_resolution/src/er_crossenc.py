"""Cross-encoder pair scorer: reads (S1 record, candidate record) jointly and outputs P(match).

Used as an extra feature for the stage-2 LightGBM. Base model: intfloat/multilingual-e5-small (MIT, 118M) with a
1-logit classification head, trained with BCEWithLogitsLoss on stage-1 candidates (positives + hard negatives).

    python -m src.er_crossenc train --chunks runs/E007/train_chunks --out runs/E008-ce/model --n 2000000
    python -m src.er_crossenc score --model runs/E008-ce/model --chunks runs/E007/test_chunks --split test \\
        --out runs/E008-ce/test_ce

Training is resumable (``<out>/ckpt/last.pt``, atomic save) and scoring skips chunks whose output exists.
Records are raw text (no transliteration); nothing branches on country. The base model is loaded from the local
Hugging Face cache (``scripts/prefetch_models.py``); the pipeline itself makes no network calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
NEG_WEIGHT_FLOOR = 0.05   # hard-negative weight = prob + floor, so easy negatives still appear occasionally


# ----------------------------------------------------------------------------------------------------- text / data
def record_text(name: str, address: str) -> str:
    """Model input for one record: raw name and address joined by ' | '."""
    return f"{name} | {address}"


def _texts(df: pd.DataFrame) -> pd.Series:
    """record_text for every row of a source frame indexed by entity_id."""
    return pd.Series([record_text(n, a) for n, a in zip(df["business_name"], df["business_address"])],
                     index=df.index, dtype=object)


def _pair_texts(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(S1 text, candidate text) arrays aligned with pairs; raises if an id is missing from left/right."""
    out = []
    for name, src in (("s1_id", left), ("cand_id", right)):
        ids = np.asarray(pairs[name], dtype=object)
        sub = src.reindex(pd.unique(ids))           # text only for the records these pairs use (not all ~12M)
        miss = sub["business_name"].isna()
        if miss.any():
            raise KeyError(f"{int(miss.sum())} {name} values not found in source records")
        out.append(_texts(sub).reindex(ids).to_numpy())
    return out[0], out[1]


def sample_training_pairs(chunks: pd.DataFrame, n: int, pos_frac: float = 0.4, seed: int = 0) -> pd.DataFrame:
    """Sample ~n pairs: up to n*pos_frac positives, the rest negatives weighted toward high stage-1 prob.

    Negatives are drawn without replacement with weight ``prob + NEG_WEIGHT_FLOOR`` (Efraimidis-Spirakis keys,
    O(N)). If there are too few positives the remainder is filled with negatives. Returns a shuffled frame with
    columns s1_id, cand_id (str) and y (bool).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(chunks["y"], dtype=bool)
    prob = np.nan_to_num(np.asarray(chunks["prob"], dtype=np.float64), nan=0.0).clip(0.0, 1.0)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    n_pos = min(len(pos), int(round(n * pos_frac)))
    n_neg = min(len(neg), n - n_pos)
    pos_pick = rng.choice(pos, size=n_pos, replace=False) if n_pos < len(pos) else pos
    if n_neg < len(neg):
        keys = np.log(rng.random(len(neg))) / (prob[neg] + NEG_WEIGHT_FLOOR)   # larger = more likely kept
        neg_pick = neg[np.argpartition(-keys, n_neg - 1)[:n_neg]] if n_neg > 0 else neg[:0]
    else:
        neg_pick = neg
    idx = rng.permutation(np.concatenate([pos_pick, neg_pick]))
    return pd.DataFrame({
        "s1_id": np.asarray(chunks["s1_id"], dtype=object)[idx].astype(str),
        "cand_id": np.asarray(chunks["cand_id"], dtype=object)[idx].astype(str),
        "y": y[idx],
    })


def _pairs_fingerprint(pairs: pd.DataFrame) -> str:
    """Stable hash of the training pairs, so a resume never continues on a different sample."""
    h = pd.util.hash_pandas_object(pairs[["s1_id", "cand_id", "y"]].astype(str), index=False).to_numpy()
    return hashlib.sha1(h.tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------ torch bits
def _device(device):
    """Resolve device=None to cuda if available else cpu."""
    import torch

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _autocast(device):
    """bf16 autocast on CUDA, no-op elsewhere."""
    import torch

    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _prefetch(fn, items: list, workers: int = 1) -> Iterator:
    """Yield fn(item) in order while the next item is prepared in a background thread (tokenizer overlap)."""
    if not items:
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = ex.submit(fn, items[0])
        for nxt in items[1:]:
            cur, fut = fut, ex.submit(fn, nxt)
            yield cur.result()
        yield fut.result()


def _save_ckpt(path: Path, **state) -> None:
    """Atomic checkpoint save: tmp file + os.replace, so a crash never corrupts last.pt."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------------- train
def train(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, out_dir, model_name: str = DEFAULT_MODEL,
          epochs: int = 1, batch: int = 128, lr: float = 3e-5, max_len: int = 96, device=None,
          warmup_frac: float = 0.06, weight_decay: float = 0.01, ckpt_every: int = 1000, seed: int = 0,
          max_steps: int | None = None, run_id: str | None = None) -> Path:
    """Fine-tune a 1-logit cross-encoder on labelled pairs; save model + tokenizer to out_dir and return it.

    left/right: source frames indexed by entity_id with business_name, business_address. Resumes from
    ``out_dir/ckpt/last.pt`` if present (same pairs required). ``max_steps`` stops early after checkpointing
    (smoke runs / tests); a completed run writes ``train_meta.json`` and is skipped on rerun.
    """
    import torch
    from torch import nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from monitor import Heartbeat
    from src.seed import seed_everything

    out_dir = Path(out_dir)
    meta_path = out_dir / "train_meta.json"
    if meta_path.exists():
        print(f"cross-encoder already trained: {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "ckpt" / "last.pt"
    dev = _device(device)
    seed_everything(seed)

    text_a, text_b = _pair_texts(pairs, left, right)
    labels = np.asarray(pairs["y"], dtype=np.float32)
    fp = _pairs_fingerprint(pairs)
    n = len(labels)
    steps_per_epoch = math.ceil(n / batch)
    total = epochs * steps_per_epoch
    warmup = int(total * warmup_frac)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(dev)
    decay = [p for nm, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for nm, p in model.named_parameters() if p.ndim < 2]   # biases, LayerNorm
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)

    def lr_lambda(s: int) -> float:
        """Linear warmup then linear decay to 0."""
        if s < warmup:
            return (s + 1) / (warmup + 1)
        return max(0.0, (total - s) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step, resumed_from = 0, None
    if ckpt.exists():
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        if s["fingerprint"] != fp or s["total"] != total:
            raise ValueError(f"{ckpt} was written for different pairs/settings; delete it to restart")
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step = resumed_from = s["step"]
        torch.set_rng_state(s["rng_cpu"].cpu())
        if dev.type == "cuda" and s.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(s["rng_cuda"].cpu())
        print(f"resumed from {ckpt} at step {step}/{total}")

    def save(at: int) -> None:
        """Checkpoint everything needed to resume exactly at step `at`."""
        _save_ckpt(ckpt, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=at,
                   total=total, fingerprint=fp, rng_cpu=torch.get_rng_state(),
                   rng_cuda=torch.cuda.get_rng_state() if dev.type == "cuda" else None)

    def encode(ix: np.ndarray):
        """Tokenize one batch of pairs (runs in the prefetch thread)."""
        enc = tok(list(text_a[ix]), list(text_b[ix]), truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt")
        return enc, torch.from_numpy(labels[ix])

    loss_fn = nn.BCEWithLogitsLoss()
    stop_at = total if max_steps is None else min(total, max_steps)
    rid = run_id or os.environ.get("RUN_ID") or f"crossenc-{out_dir.name}"
    t0 = time.time()
    with Heartbeat(rid, total_steps=total, metric_name="loss", higher_is_better=False,
                   meta={"model": model_name, "pairs": n, "out_dir": str(out_dir)}) as hb:
        hb.step(step, force=True)
        model.train()
        while step < stop_at:
            epoch = step // steps_per_epoch
            perm = np.random.default_rng(seed + epoch).permutation(n)   # same order on resume
            first = step - epoch * steps_per_epoch
            last = min(steps_per_epoch, first + stop_at - step)
            batches = [perm[i * batch:(i + 1) * batch] for i in range(first, last)]
            for enc, yb in _prefetch(encode, batches):
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                yb = yb.to(dev, non_blocking=True)
                with _autocast(dev):
                    logits = model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                hb.step(step, loss=loss.item(), epoch=step / steps_per_epoch, lr=sched.get_last_lr()[0],
                        samples=len(yb))
                if step % ckpt_every == 0 and step < stop_at:
                    save(step)
        if step < total:                       # stopped early via max_steps: leave a resumable checkpoint
            save(step)
            print(f"stopped at step {step}/{total}; rerun to resume")
            return out_dir
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    meta_path.write_text(json.dumps({"base_model": model_name, "pairs": n, "positives": int(labels.sum()),
                                     "epochs": epochs, "batch": batch, "lr": lr, "max_len": max_len,
                                     "steps": total, "seed": seed, "resumed_from_step": resumed_from,
                                     "fingerprint": fp, "runtime_s": round(time.time() - t0, 1)}, indent=2),
                         encoding="utf-8")
    print(f"saved cross-encoder to {out_dir} ({total} steps, {time.time() - t0:.0f}s)")
    return out_dir


# --------------------------------------------------------------------------------------------------------- score
def load_model(model_dir, device=None):
    """Load (model, tokenizer, device) from a directory written by train()."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    return model, tok, dev


def score_loaded(model, tok, dev, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                 batch: int = 512, max_len: int = 96) -> np.ndarray:
    """P(match) for pairs with an already-loaded model (see score)."""
    import torch

    text_a, text_b = _pair_texts(pairs, left, right)
    out = np.empty(len(text_a), dtype=np.float32)
    if len(out) == 0:
        return out
    lengths = np.fromiter((len(x) + len(y) for x, y in zip(text_a, text_b)), dtype=np.int64, count=len(text_a))
    order = np.argsort(lengths, kind="stable")                  # length-sorted batches -> minimal padding
    batches = [order[i:i + batch] for i in range(0, len(order), batch)]

    def encode(ix: np.ndarray):
        """Tokenize one batch (runs in the prefetch thread)."""
        return ix, tok(list(text_a[ix]), list(text_b[ix]), truncation=True, max_length=max_len, padding=True,
                       return_tensors="pt")

    with torch.inference_mode():
        for ix, enc in _prefetch(encode, batches):
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            with _autocast(dev):
                logits = model(**enc).logits.squeeze(-1)
            out[ix] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def score(model_dir, pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, batch: int = 512,
          max_len: int = 96, device=None) -> np.ndarray:
    """Return float32 P(match) for each row of pairs (same order), using the model saved in model_dir."""
    model, tok, dev = load_model(model_dir, device)
    return score_loaded(model, tok, dev, pairs, left, right, batch=batch, max_len=max_len)


# ----------------------------------------------------------------------------------------------------------- CLI
def load_sources(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(left, right) for a split from data/cache: S1 and concat(S2, S3), indexed by entity_id."""
    from src.er_data import load

    s1, s2, s3, _ = load(split, with_gt=False)
    cols = ["business_name", "business_address"]
    left = s1.set_index("entity_id")[cols]
    right = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")[cols]
    return left, right


def chunk_files(chunks_dir) -> list[Path]:
    """Sorted parquet chunk files in a directory (raises if there are none)."""
    files = sorted(Path(chunks_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet chunks in {chunks_dir}")
    return files


def read_chunks(chunks_dir, columns: list[str]) -> pd.DataFrame:
    """Concatenate the given columns of every chunk (arrow-backed strings keep 30M+ rows in memory cheaply)."""
    return pd.concat([pd.read_parquet(p, columns=columns, dtype_backend="pyarrow") for p in chunk_files(chunks_dir)],
                     ignore_index=True)


def excluded_s1(folds: list[int]) -> set[str]:
    """S1 ids in the given folds of data/cache/folds_s1_k5.parquet (the shared S1 fold file)."""
    from src.er_data import cache_dir

    f = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    return set(f.loc[f.fold.isin(folds), "s1_id"].astype(str))


def cmd_train(a: argparse.Namespace) -> None:
    """CLI: sample pairs once (cached in <out>/train_pairs.parquet), then train (resumable)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "train_pairs.parquet"
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
        print(f"reusing {len(pairs):,} sampled pairs from {pairs_path}")
    else:
        ch = read_chunks(a.chunks, ["s1_id", "cand_id", "prob", "y"])
        if a.exclude_folds:
            drop = excluded_s1(a.exclude_folds)
            ch = ch[~ch.s1_id.isin(list(drop))].reset_index(drop=True)
            print(f"excluded folds {a.exclude_folds}: {len(drop):,} S1 dropped, {len(ch):,} pairs left")
        pairs = sample_training_pairs(ch, a.n, pos_frac=a.pos_frac, seed=a.seed)
        del ch
        tmp = pairs_path.with_suffix(".tmp")
        pairs.to_parquet(tmp, index=False)
        os.replace(tmp, pairs_path)
        print(f"sampled {len(pairs):,} pairs ({pairs.y.mean():.1%} positive) -> {pairs_path}")
    left, right = load_sources(a.split)
    train(pairs, left, right, out, model_name=a.model_name, epochs=a.epochs, batch=a.batch, lr=a.lr,
          max_len=a.max_len, device=a.device, ckpt_every=a.ckpt_every, seed=a.seed, max_steps=a.max_steps)


def cmd_score(a: argparse.Namespace) -> None:
    """CLI: write <out>/<chunk>.parquet [s1_id, cand_id, ce_score] per input chunk; skips finished chunks."""
    from monitor import Heartbeat

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = chunk_files(a.chunks)
    todo = [p for p in files if not (out / p.name).exists()]
    print(f"{len(files) - len(todo)}/{len(files)} chunks already scored")
    if not todo:
        return
    left, right = load_sources(a.split)
    model, tok, dev = load_model(a.model, a.device)
    t0 = time.time()
    with Heartbeat(os.environ.get("RUN_ID") or f"crossenc-score-{a.split}", total_steps=len(files),
                   every_steps=1, meta={"model": str(a.model), "chunks": str(a.chunks)}) as hb:
        keep = excluded_s1(a.only_folds) if a.only_folds else None   # S1s whose pairs we actually need
        for p in todo:
            pairs = pd.read_parquet(p, columns=["s1_id", "cand_id"] + (["prob"] if a.keep_prob else []))
            scores = np.full(len(pairs), np.nan, dtype=np.float32)   # rows stay aligned with the chunk
            sel = np.ones(len(pairs), bool) if keep is None else pairs.s1_id.isin(keep).to_numpy()
            if a.keep_prob:
                # only the final candidate rows (stage-1 prob >= keep_prob OR filter CE >= keep_ce), as in
                # er_fullpass.prune_rows; the filter CE's files are row-aligned with the chunks (same names)
                cand = pairs.pop("prob").to_numpy() >= a.keep_prob
                if a.keep_ce_dir:
                    ce = pd.read_parquet(Path(a.keep_ce_dir) / f"{a.split}_ce" / p.name)
                    if len(ce) != len(pairs) or not (ce.cand_id.to_numpy() == pairs.cand_id.to_numpy()).all():
                        raise ValueError(f"{a.keep_ce_dir} rows misaligned with {p.name}")
                    cand |= np.nan_to_num(ce.ce_score.to_numpy(np.float32), nan=0.0) >= a.keep_ce
                sel &= cand
                print(f"{p.name}: scoring {sel.sum():,} of {len(pairs):,} rows (final candidate filter)")
            if sel.any():
                scores[sel] = score_loaded(model, tok, dev, pairs[sel].reset_index(drop=True), left, right,
                                           batch=a.batch, max_len=a.max_len)
            pairs["ce_score"] = scores
            tmp = out / (p.name + ".tmp")
            pairs.to_parquet(tmp, index=False)
            os.replace(tmp, out / p.name)
            done = files.index(p) + 1
            hb.step(done, force=True, samples=len(pairs), chunk=p.name)
            print(f"{p.name}: {len(pairs):,} pairs [{time.time() - t0:.0f}s]")


def main(argv: list[str] | None = None) -> None:
    """Entry point: `train` or `score` subcommand."""
    ap = argparse.ArgumentParser(prog="python -m src.er_crossenc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="sample pairs from stage-1 chunks and fine-tune the cross-encoder")
    t.add_argument("--chunks", required=True, help="dir of stage-1 chunk parquets (s1_id, cand_id, prob, y)")
    t.add_argument("--out", required=True)
    t.add_argument("--n", type=int, default=2_000_000)
    t.add_argument("--split", default="train")
    t.add_argument("--pos-frac", type=float, default=0.4)
    t.add_argument("--exclude-folds", type=int, nargs="*", default=[],
                   help="drop S1s in these folds of folds_s1_k5.parquet (e.g. 0 to keep dev unseen)")
    t.add_argument("--model-name", default=DEFAULT_MODEL)
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--max-len", type=int, default=96)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--max-steps", type=int, default=None, help="stop early (smoke test); rerun resumes")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None)
    t.set_defaults(fn=cmd_train)
    s = sub.add_parser("score", help="score every chunk with a trained cross-encoder")
    s.add_argument("--model", required=True)
    s.add_argument("--chunks", required=True)
    s.add_argument("--split", required=True, choices=["train", "test"])
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--max-len", type=int, default=96)
    s.add_argument("--device", default=None)
    s.add_argument("--only-folds", type=int, nargs="*", default=[],
                   help="score only S1s in these folds (others get NaN, rows stay aligned); e.g. 0 3 4")
    s.add_argument("--keep-prob", type=float, default=0.0,
                   help="score only rows with stage-1 prob >= this OR --keep-ce-dir score >= --keep-ce (others NaN)")
    s.add_argument("--keep-ce-dir", default="", help="filter CE run dir with <split>_ce/ (e.g. runs/E016-ce)")
    s.add_argument("--keep-ce", type=float, default=0.0)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
