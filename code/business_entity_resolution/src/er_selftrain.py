"""Self-training the cross-encoder on confident test pseudo-labels for one country (France: unseen in train).

Why (sub-09 test predictions): France has 4.65% of its candidate pairs in the uncertain band (prob 0.1-0.9) vs ~1.1%
for US/India, and it is 15% of test. Its earlier LB errors were false merges. Continuing to train the cross-encoder on
France's CONFIDENT pairs (pos: stage-2 prob >= pos_thr, hard neg: prob <= neg_thr, weighted by stage-1 prob) teaches
French spellings/abbreviations the model never saw, which then transfers to the uncertain French pairs. Allowed: test
records are provided data (docs/COMPETITION.md, confirmed 2026-09-26); no labels or external data are used.

    python -m src.er_selftrain train --probs runs/E015/sub_E016/test_probs_stage2.parquet --country France \\
        --base runs/E016-ce-base/model --out runs/E018-ce-fr/model
    python -m src.er_selftrain score --model runs/E018-ce-fr/model --country France \\
        --base-ce runs/E016-ce --out runs/E018-ce          # E016 scores, French test rows replaced; train_ce copied

The country is a parameter (open set of strings), never hard-coded in the logic.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.er_crossenc import load_model, load_sources, score_loaded, train

NEG_FLOOR = 0.05


def pseudo_pairs(probs: pd.DataFrame, chunks_dir: str, country_of: pd.Series, country, pos_thr: float,
                 neg_thr: float, n_pos: int, n_neg: int, seed: int = 0) -> pd.DataFrame:
    """Confident pseudo-labelled test pairs of the given country (or list of countries): [s1_id, cand_id, y].
    Negatives favour high stage-1 prob."""
    countries = [country] if isinstance(country, str) else list(country)
    rng = np.random.default_rng(seed)
    p = probs[np.isin(country_of.reindex(probs.s1_id).to_numpy(), countries)]
    s1p = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "prob"]).rename(columns={"prob": "p1"})
                     for f in sorted(Path(chunks_dir).glob("*.parquet"))], ignore_index=True)
    p = p.merge(s1p, on=["s1_id", "cand_id"], how="left")
    pos = p[p.prob >= pos_thr]
    neg = p[p.prob <= neg_thr]
    pos = pos.sample(min(n_pos, len(pos)), random_state=seed)
    if len(neg) > n_neg:
        keys = np.log(rng.random(len(neg))) / (neg.p1.fillna(0).clip(0, 1).to_numpy() + NEG_FLOOR)
        neg = neg.iloc[np.argpartition(-keys, n_neg - 1)[:n_neg]]
    out = pd.concat([pos.assign(y=True), neg.assign(y=False)])[["s1_id", "cand_id", "y"]]
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prefixed_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(left, right) for train + test with ids prefixed 'tr:' / 'te:' (entity ids may repeat across splits)."""
    lt, rt = load_sources("train")
    le, re_ = load_sources("test")
    left = pd.concat([lt.rename(index=lambda x: "tr:" + x), le.rename(index=lambda x: "te:" + x)])
    right = pd.concat([rt.rename(index=lambda x: "tr:" + x), re_.rename(index=lambda x: "te:" + x)])
    return left, right


def cmd_train(a: argparse.Namespace) -> None:
    """Continue training the base cross-encoder on country pseudo-labels mixed with original train pairs."""
    from src.er_data import load

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "selftrain_pairs.parquet"
    if pairs_path.exists():
        pairs = pd.read_parquet(pairs_path)
    else:
        s1 = load("test", with_gt=False)[0]
        country_of = s1.set_index("entity_id").country
        pl = pseudo_pairs(pd.read_parquet(a.probs), a.chunks, country_of, a.country, a.pos_thr, a.neg_thr,
                          a.n_pos, a.n_neg, a.seed)
        pl = pl.assign(s1_id="te:" + pl.s1_id, cand_id="te:" + pl.cand_id)
        orig = pd.read_parquet(Path(a.base) / "train_pairs.parquet")
        orig = orig.sample(min(int(len(pl) * a.orig_frac), len(orig)), random_state=a.seed)
        orig = orig.assign(s1_id="tr:" + orig.s1_id, cand_id="tr:" + orig.cand_id)
        pairs = pd.concat([pl, orig[["s1_id", "cand_id", "y"]]]).sample(frac=1.0, random_state=a.seed)
        pairs = pairs.reset_index(drop=True)
        pairs.to_parquet(pairs_path, index=False)
        print(f"pseudo pairs {len(pl):,} ({pl.y.mean():.1%} pos, {' '.join(a.country)}) + {len(orig):,} original train pairs")
    left, right = prefixed_sources()
    train(pairs, left, right, out, model_name=a.base, epochs=1, batch=a.batch, lr=a.lr, max_len=96,
          ckpt_every=2000, seed=a.seed)


def cmd_score(a: argparse.Namespace) -> None:
    """Base CE scores for every test chunk, with the given country's rows re-scored by the adapted model."""
    from monitor import Heartbeat
    from src.er_data import load

    out, base = Path(a.out), Path(a.base_ce)
    (out / "test_ce").mkdir(parents=True, exist_ok=True)
    if not (out / "train_ce").exists():
        shutil.copytree(base / "train_ce", out / "train_ce")    # train side unchanged (dev/fit rows are US/India)
    s1 = load("test", with_gt=False)[0]
    country_of = s1.set_index("entity_id").country
    files = sorted((base / "test_ce").glob("*.parquet"))
    left, right = load_sources("test")
    model, tok, dev = load_model(a.model, a.device)
    t0 = time.time()
    with Heartbeat(os.environ.get("RUN_ID") or "selftrain-score", total_steps=len(files), every_steps=1) as hb:
        for i, f in enumerate(files):
            if (out / "test_ce" / f.name).exists():
                continue
            ce = pd.read_parquet(f)
            sel = np.isin(country_of.reindex(ce.s1_id).to_numpy(), a.country)
            if a.only_scored:   # base CE scored only the final candidate rows (NaN elsewhere): keep that support
                sel &= ce.ce_score.notna().to_numpy()
            if a.uncertain:     # only pairs the base CE is unsure about; confident ones keep the base score
                sel &= ((ce.ce_score > a.uncertain[0]) & (ce.ce_score < a.uncertain[1])).to_numpy()
            if sel.any():
                ce.loc[sel, "ce_score"] = score_loaded(model, tok, dev, ce.loc[sel, ["s1_id", "cand_id"]].reset_index(drop=True),
                                                       left, right, batch=a.batch)
            tmp = out / "test_ce" / (f.name + ".tmp")
            ce.to_parquet(tmp, index=False)
            os.replace(tmp, out / "test_ce" / f.name)
            hb.step(i + 1, force=True)
            print(f"{f.name}: {int(sel.sum()):,} {' '.join(a.country)} rows re-scored [{time.time() - t0:.0f}s]", flush=True)


def main() -> None:
    """CLI entry point: train / score subcommands."""
    ap = argparse.ArgumentParser(prog="python -m src.er_selftrain")
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--probs", required=True, help="stage-2 test probabilities (s1_id, cand_id, prob)")
    t.add_argument("--chunks", default="runs/E015/test_chunks", help="stage-1 test chunks (for hard-negative weights)")
    t.add_argument("--country", nargs="+", required=True)
    t.add_argument("--base", required=True, help="cross-encoder dir to continue from (has train_pairs.parquet)")
    t.add_argument("--out", required=True)
    t.add_argument("--pos-thr", type=float, default=0.97)
    t.add_argument("--neg-thr", type=float, default=0.03)
    t.add_argument("--n-pos", type=int, default=250_000)
    t.add_argument("--n-neg", type=int, default=350_000)
    t.add_argument("--orig-frac", type=float, default=0.33, help="original train pairs per pseudo pair")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--country", nargs="+", required=True)
    s.add_argument("--uncertain", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="re-score only rows whose base CE score is in (LO, HI), e.g. 0.02 0.98")
    s.add_argument("--only-scored", action="store_true", help="re-score only rows the base CE scored (not NaN)")
    s.add_argument("--base-ce", required=True, help="CE score dir to copy (train_ce) and patch (test_ce)")
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--device", default=None)
    a = ap.parse_args()
    {"train": cmd_train, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
