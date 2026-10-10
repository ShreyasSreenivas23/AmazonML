"""Reference training loop: K-fold, bf16, grad accumulation, resumable checkpoints, heartbeat, OOF.

Runs on synthetic data so it works anywhere (cloud VM without GPU included). Copy it for
real models; keep the resume/heartbeat/OOF skeleton, swap data + model.

    python -m monitor.launch --run E000-template -- python -m src.train_template --cfg configs/example.yaml
    python -m src.train_template --cfg configs/example.yaml train.epochs=3     # overrides
Kill it mid-run and rerun the same command: it resumes from CKPT_DIR/<exp_id>/f<k>/last.pt.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from monitor import Heartbeat
from src.config import get_paths, load_config, save_config
from src.cv import make_folds
from src.metrics import get_metric
from src.oof import save_oof
from src.seed import seed_everything


def synthetic(n: int = 4000, d: int = 32, seed: int = 0):
    """Small synthetic regression dataset for smoke tests."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32")
    y = np.exp(X[:, :4].sum(1) * 0.3 + rng.randn(n) * 0.1).astype("float32")  # positive, skewed (price-like)
    return X, y


def save_ckpt(path: Path, **state) -> None:
    """Atomically save a training checkpoint (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save never corrupts last.pt


def train_fold(cfg, k, X, y, tr, va, device, metric_fn, hb: Heartbeat) -> np.ndarray:
    """Train one fold with checkpointing and heartbeats; returns OOF and test predictions."""
    ckpt_dir = get_paths().ckpt / cfg.exp_id / f"f{k}"
    pred_path = ckpt_dir / "val_pred.npy"
    if pred_path.exists():
        print(f"fold {k}: done, skipping")
        return np.load(pred_path)

    y_t = np.log1p(y)  # target transform matched to a relative-error metric; invert before scoring
    dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y_t[tr])),
                    batch_size=cfg.train.batch_size, shuffle=True, drop_last=True,
                    num_workers=0, pin_memory=device.type == "cuda")
    model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    total = cfg.train.epochs * len(dl) // cfg.train.grad_accum
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.lr, total_steps=max(total, 1))
    step0 = k * total  # global step offset so one heartbeat covers all folds
    step, start_epoch, best = 0, 0, None
    hb.reset_best()
    hb.state.setdefault("extra", {})["fold"] = k

    last = ckpt_dir / "last.pt"
    if last.exists():
        s = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step, start_epoch, best = s["step"], s["epoch"] + 1, s["best"]
        torch.set_rng_state(s["rng_cpu"].cpu())  # map_location moved it to GPU
        print(f"fold {k}: resumed at epoch {start_epoch}, step {step}")

    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.train.precision == "bf16")
    Xv = torch.from_numpy(X[va]).to(device)
    hb.step(step0 + step, force=True)
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        for i, (xb, yb) in enumerate(dl):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.autocast(**amp):
                loss = nn.functional.mse_loss(model(xb).squeeze(1).float(), yb) / cfg.train.grad_accum
            loss.backward()
            if (i + 1) % cfg.train.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                hb.step(step0 + step, loss=loss.item() * cfg.train.grad_accum, epoch=epoch,
                        lr=sched.get_last_lr()[0], samples=len(xb) * cfg.train.grad_accum)
        model.eval()
        with torch.no_grad(), torch.autocast(**amp):
            pred = np.expm1(model(Xv).squeeze(1).float().cpu().numpy()).clip(0)
        score = metric_fn(y[va], pred)
        hb.val(score)
        if best is None or (score > best if hb.higher_is_better else score < best):  # vs resumed best too
            best = score
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            np.save(ckpt_dir / "best_val_pred.npy", pred)
        print(f"fold {k} epoch {epoch}: {cfg.metric}={score:.4f} best={best:.4f}")
        save_ckpt(last, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  step=step, epoch=epoch, best=best, rng_cpu=torch.get_rng_state())
    pred = np.load(ckpt_dir / "best_val_pred.npy")
    np.save(pred_path, pred)
    return pred


def main(argv: list[str] | None = None) -> float:
    """CLI: resumable k-fold training template."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/example.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = load_config(a.cfg, a.overrides)
    seed_everything(cfg.seed)
    run_dir = get_paths().runs / os.environ.get("RUN_ID", cfg.exp_id)
    save_config(cfg, run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_fn, higher = get_metric(cfg.metric)

    import pandas as pd

    X, y = synthetic(seed=cfg.seed)
    df = pd.DataFrame({"id": np.arange(len(y)), "y": y})
    folds = make_folds(df, cfg.cv.n_splits, "stratified_reg", target="y", seed=cfg.seed)
    oof = np.zeros(len(y), dtype="float64")
    t0 = time.time()
    n_tr = int(len(y) * (1 - 1 / cfg.cv.n_splits))
    steps_per_fold = cfg.train.epochs * (n_tr // cfg.train.batch_size) // cfg.train.grad_accum
    # ONE heartbeat per process, named after the launcher's RUN_ID so monitor.watch finds it
    with Heartbeat(os.environ.get("RUN_ID", cfg.exp_id), total_steps=steps_per_fold * cfg.cv.n_splits,
                   higher_is_better=higher, metric_name=cfg.metric, meta={"exp_id": cfg.exp_id},
                   every_steps=cfg.monitor.heartbeat_every_steps,
                   every_seconds=cfg.monitor.heartbeat_every_seconds) as hb:
        for k in range(cfg.cv.n_splits):
            tr, va = np.where(folds.fold != k)[0], np.where(folds.fold == k)[0]
            oof[va] = train_fold(cfg, k, X, y, tr, va, device, metric_fn, hb)
        fold_scores = [metric_fn(y[folds.fold == k], oof[folds.fold == k]) for k in range(cfg.cv.n_splits)]
        cv = metric_fn(y, oof)
        hb.reset_best()
        hb.val(cv, cv_std=float(np.std(fold_scores)))  # final "best" reported by the watchdog = overall CV
    d = save_oof(cfg.exp_id, df["id"], oof, folds["fold"], metric=cfg.metric, cv=cv, fold_scores=fold_scores,
                 runtime_s=round(time.time() - t0, 1))
    print(f"CV {cfg.metric} = {cv:.4f} +/- {np.std(fold_scores):.4f} | OOF -> {d}")
    return cv


if __name__ == "__main__":
    main()
