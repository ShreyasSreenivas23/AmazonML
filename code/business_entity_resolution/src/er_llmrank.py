"""LLM reranker (Qwen3-4B + LoRA) for the uncertain pairs, with competing-S1 context in the prompt (E021).

Why (E015/E017 dev error analysis): ~85% of the remaining loss is true matches that are retrieved but rejected, mostly
empty-address records whose name nearly matches several S1s. A pair model (our cross-encoder) never sees the
competitors; this prompt shows the S1, the candidate record AND the other S1s that also retrieved the record, and asks
a 4B multilingual LLM (Apache-2.0, <= 7B team cap; docs/COMPETITION.md) for Yes/No. Scored only on the uncertain band
(E016 CE score in (lo, hi)), identically for fit, dev and test, and used as one more stage-2 feature.

    python -m src.er_llmrank build --out runs/E021-llm/data           # CPU: prompts for train / fit+dev / test
    python -m src.er_llmrank train --data runs/E021-llm/data --out runs/E021-llm/lora     # GPU (AWS L40S or local)
    python -m src.er_llmrank score --data runs/E021-llm/data --lora runs/E021-llm/lora --split test --out runs/E021-llm
    python -m src.er_llmrank to-ce --llm runs/E021-llm --out runs/E021-llmce    # stage-2 --ce-dir format (NaN outside band)

Training pairs come from S1 folds 1-4 only (fold 0 = stage-2 fit pool + dev stays unseen). Competitors are taken
from the full population of the same split (stage-1 chunks), so train and test prompts mean the same thing.
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

BASE = "Qwen/Qwen3-4B"
N_COMP = 3


def _clip(s: str, n: int) -> str:
    """Collapse whitespace and truncate a field to n characters for the prompt."""
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def prompt(s1_name, s1_addr, country, c_name, c_addr, competitors: list[tuple[str, str]]) -> str:
    """One Yes/No question; the answer token is appended at training time."""
    comp = "\n".join(f"  - {_clip(n, 60)} | {_clip(a, 70) or '(no address)'}" for n, a in competitors) or "  (none)"
    return (f"Business A ({country}): {_clip(s1_name, 80)} | {_clip(s1_addr, 90) or '(no address)'}\n"
            f"Record B: {_clip(c_name, 80)} | {_clip(c_addr, 90) or '(no address)'}\n"
            f"Other businesses that record B also resembles:\n{comp}\n"
            f"Is record B the same real-world business as business A? Answer Yes or No.\nAnswer:")


# ----------------------------------------------------------------------------------------------------------- build
def competitors_of(chunks: pd.DataFrame, k: int = N_COMP + 1) -> pd.Series:
    """cand_id -> list of its top-k S1s by stage-1 prob over the whole population."""
    c = chunks[["s1_id", "cand_id", "prob"]].sort_values(["cand_id", "prob"], ascending=[True, False], kind="stable")
    c = c[c.groupby("cand_id", sort=False).cumcount() < k]
    return c.groupby("cand_id", sort=False).s1_id.agg(list)


def prompts_for(pairs: pd.DataFrame, comp: pd.Series, left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Prompt per pair; competitors exclude the pair's own S1."""
    out = []
    L, R = left, right
    for s1, c in zip(pairs.s1_id.to_numpy(), pairs.cand_id.to_numpy()):
        others = [o for o in comp.get(c, []) if o != s1][:N_COMP]
        comps = [(L.at[o, "business_name"], L.at[o, "business_address"]) for o in others]
        out.append(prompt(L.at[s1, "business_name"], L.at[s1, "business_address"], L.at[s1, "country"],
                          R.at[c, "business_name"], R.at[c, "business_address"], comps))
    return out


def _sources(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw S1 and S2+S3 frames of a split, indexed by entity_id."""
    from src.er_data import load
    s1, s2, s3, _ = load(split, with_gt=False)
    return s1.set_index("entity_id"), pd.concat([s2, s3], ignore_index=True).set_index("entity_id")


def _chunks(d: str, cols: list[str]) -> pd.DataFrame:
    """Concatenate the given columns of every parquet in a folder."""
    return pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(Path(d).glob("*.parquet"))], ignore_index=True)


def _band(ce_dir: str, split: str, lo: float, hi: float) -> pd.DataFrame:
    """Pairs whose cross-encoder score lies in (lo, hi): the uncertain band the LLM scores."""
    ce = _chunks(f"{ce_dir}/{split}_ce", ["s1_id", "cand_id", "ce_score"])
    return ce[(ce.ce_score > lo) & (ce.ce_score < hi)][["s1_id", "cand_id"]].reset_index(drop=True)


def cmd_build(a: argparse.Namespace) -> None:
    """Build training prompts (folds 1-4) and scoring prompts (fold-0 and test band) with competitor context."""
    from src.er_data import cache_dir

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet").set_index("s1_id")
    rng = np.random.default_rng(a.seed)
    # ---- train split: LoRA training pairs (folds 1-4) + fold-0 band rows to score (stage-2 fit pool + dev)
    ch = _chunks(f"{a.exp_dir}/train_chunks", ["s1_id", "cand_id", "prob", "y"])
    comp = competitors_of(ch)
    left, right = _sources("train")
    f = folds.fold.reindex(ch.s1_id).to_numpy()
    tr = ch[f != 0]
    hard = tr[(tr.prob > a.lo) & (tr.prob < a.hi)]
    easy = tr[~((tr.prob > a.lo) & (tr.prob < a.hi))]
    n_hard = min(len(hard), int(a.n_train * 0.8))
    pick = pd.concat([hard.sample(n_hard, random_state=a.seed),
                      easy.sample(a.n_train - n_hard, random_state=a.seed, weights=easy.prob + 0.02)])
    pos, neg = pick[pick.y], pick[~pick.y]
    n_pos = min(len(pos), int(a.n_train * a.pos_frac))
    pick = pd.concat([pos.sample(n_pos, random_state=a.seed),
                      neg.sample(min(len(neg), a.n_train - n_pos), random_state=a.seed)]).sample(frac=1.0, random_state=a.seed)
    trn = pick[["s1_id", "cand_id", "y"]].reset_index(drop=True)
    trn["prompt"] = prompts_for(trn, comp, left, right)
    trn.to_parquet(out / "train.parquet", index=False)
    print(f"train prompts {len(trn):,} ({trn.y.mean():.1%} pos, {n_hard / len(trn):.0%} from the hard band)", flush=True)
    band = _band(a.ce_dir, "train", a.lo, a.hi)
    band = band[folds.fold.reindex(band.s1_id).to_numpy() == 0].reset_index(drop=True)
    band["prompt"] = prompts_for(band, comp, left, right)
    band.to_parquet(out / "score_train.parquet", index=False)
    print(f"fold-0 band prompts {len(band):,}", flush=True)
    del ch, comp, left, right
    # ---- test split
    ch = _chunks(f"{a.exp_dir}/test_chunks", ["s1_id", "cand_id", "prob"])
    comp = competitors_of(ch)
    left, right = _sources("test")
    band = _band(a.ce_dir, "test", a.lo, a.hi)
    band["prompt"] = prompts_for(band, comp, left, right)
    band.to_parquet(out / "score_test.parquet", index=False)
    print(f"test band prompts {len(band):,}", flush=True)
    (out / "build.json").write_text(json.dumps(vars(a), indent=2), encoding="utf-8")


# ------------------------------------------------------------------------------------------------------ train/score
def _yes_no_ids(tok) -> tuple[int, int]:
    """Token ids of ' Yes' and ' No' (each must be a single token)."""
    y, n = tok.encode(" Yes", add_special_tokens=False), tok.encode(" No", add_special_tokens=False)
    assert len(y) == 1 and len(n) == 1, (y, n)
    return y[0], n[0]


def _last_logits(model, enc):
    """Logits at the last position only (LEFT padding, so it is each row's real last token).

    logits_to_keep=1 matters: full logits are vocab (151k) x tokens x batch floats, ~6 GB per batch of 64, which
    spilled the 16 GB GPU into system RAM and stalled scoring at 0% GPU (26 Sep). RoPE is relative, so left padding
    does not change attention between real tokens."""
    return model(**enc, logits_to_keep=1).logits[:, -1]


def cmd_train(a: argparse.Namespace) -> None:
    """LoRA-train the LLM to answer Yes/No at the prompt's last position (cross-entropy on the two tokens)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from monitor import Heartbeat

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(Path(a.data) / "train.parquet")
    if a.max_rows:
        df = df.head(a.max_rows)
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yes, no = _yes_no_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(df) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(df))
    y_all = torch.tensor(df.y.to_numpy().astype(np.int64))
    model.train()
    t0 = time.time()
    with Heartbeat(os.environ.get("RUN_ID", "llm-train"), total_steps=steps, every_steps=20) as hb:
        for s in range(steps):
            ix = order[s * a.batch:(s + 1) * a.batch]
            enc = tok(list(df.prompt.iloc[ix]), return_tensors="pt", padding=True, truncation=True,
                      max_length=a.max_len).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = _last_logits(model, enc)[:, [no, yes]].float()          # class 1 = Yes
            loss = torch.nn.functional.cross_entropy(lg, y_all[ix].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            hb.step(s + 1, loss=loss.item(), samples=len(ix))
            if (s + 1) % 100 == 0:
                print(f"step {s + 1}/{steps} loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
            if (s + 1) % a.ckpt_every == 0:
                model.save_pretrained(out)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({**vars(a), "steps": steps, "runtime_s": round(time.time() - t0)}),
                                         encoding="utf-8")
    print(f"saved LoRA to {out}")


def cmd_score(a: argparse.Namespace) -> None:
    """P(Yes) for every band prompt, length-sorted batches, resumable per shard."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from monitor import Heartbeat

    df = pd.read_parquet(Path(a.data) / f"score_{a.split}.parquet")
    if a.max_rows:
        df = df.head(a.max_rows)
    out = Path(a.out) / f"llm_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.lora)
    tok.padding_side = "left"
    yes, no = _yes_no_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).cuda()
    model = PeftModel.from_pretrained(model, a.lora).merge_and_unload().eval()
    lens = df.prompt.str.len().to_numpy()
    order = np.argsort(lens, kind="stable")
    shard = a.shard_rows
    n_sh = math.ceil(len(df) / shard)
    t0 = time.time()
    with Heartbeat(os.environ.get("RUN_ID", f"llm-score-{a.split}"), total_steps=n_sh, every_steps=1) as hb, \
            torch.inference_mode():
        for k in range(n_sh):
            path = out / f"{k:04d}.parquet"
            if path.exists():
                continue                                            # resumable (spot-safe)
            rows = order[k * shard:(k + 1) * shard]
            p = np.empty(len(rows), np.float32)
            for i in range(0, len(rows), a.batch):
                ix = rows[i:i + a.batch]
                enc = tok(list(df.prompt.iloc[ix]), return_tensors="pt", padding=True, truncation=True,
                          max_length=a.max_len).to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = _last_logits(model, enc)[:, [no, yes]].float()
                p[i:i + len(ix)] = torch.softmax(lg, -1)[:, 1].cpu().numpy()
            pd.DataFrame({"s1_id": df.s1_id.to_numpy()[rows], "cand_id": df.cand_id.to_numpy()[rows],
                          "llm_score": p}).to_parquet(path, index=False)
            hb.step(k + 1, force=True)
            print(f"shard {k + 1}/{n_sh} [{time.time() - t0:.0f}s]", flush=True)


def cmd_to_ce(a: argparse.Namespace) -> None:
    """Write stage-2 --ce-dir format: per stage-1 chunk, llm_score as ce_score (NaN outside the band)."""
    for split in a.splits:
        files = sorted((Path(a.llm) / f"llm_{split}").glob("*.parquet"))
        if not files:
            print(f"{split}: no llm scores yet, skipped")
            continue
        sc = pd.concat([pd.read_parquet(f) for f in files])
        key = pd.Series(sc.llm_score.to_numpy(), index=pd.MultiIndex.from_arrays([sc.s1_id, sc.cand_id]))
        od = Path(a.out) / f"{split}_ce"
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(f"{a.exp_dir}/{split}_chunks").glob("*.parquet")):
            c = pd.read_parquet(f, columns=["s1_id", "cand_id"])
            v = key.reindex(pd.MultiIndex.from_arrays([c.s1_id, c.cand_id])).to_numpy(np.float32)
            c.assign(ce_score=v).to_parquet(od / f.name, index=False)
        print(f"{split}: {len(sc):,} llm scores mapped")


def main() -> None:
    """CLI entry point: build / train / score / to-ce."""
    ap = argparse.ArgumentParser(prog="python -m src.er_llmrank")
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--exp-dir", default="runs/E015")
    b.add_argument("--ce-dir", default="runs/E016-ce", help="band is defined by this CE's scores (fit, dev, test alike)")
    b.add_argument("--lo", type=float, default=0.02)
    b.add_argument("--hi", type=float, default=0.98)
    b.add_argument("--n-train", type=int, default=120_000)
    b.add_argument("--pos-frac", type=float, default=0.4)
    b.add_argument("--seed", type=int, default=0)
    t = sp.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--base", default=BASE)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--max-len", type=int, default=256)
    t.add_argument("--ckpt-every", type=int, default=1000)
    t.add_argument("--max-rows", type=int, default=0, help="smoke test")
    t.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--lora", required=True)
    s.add_argument("--base", default=BASE)
    s.add_argument("--split", required=True, choices=["train", "test"])
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--max-len", type=int, default=256)
    s.add_argument("--shard-rows", type=int, default=50_000)
    s.add_argument("--max-rows", type=int, default=0, help="smoke test")
    c = sp.add_parser("to-ce")
    c.add_argument("--llm", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--exp-dir", default="runs/E015")
    c.add_argument("--splits", nargs="+", default=["train", "test"], help="train only = dev check before test is scored")
    a = ap.parse_args()
    {"build": cmd_build, "train": cmd_train, "score": cmd_score, "to-ce": cmd_to_ce}[a.cmd](a)


if __name__ == "__main__":
    main()
