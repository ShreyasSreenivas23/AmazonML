"""Heartbeat: a training loop writes runs/<run_id>/heartbeat.json so monitor.watch can see it.

    from monitor import Heartbeat
    with Heartbeat("E007-deberta-f0", total_steps=len(dl) * epochs, higher_is_better=False) as hb:
        for step, batch in ...:
            loss = ...
            hb.step(step, loss=loss.item(), epoch=ep, lr=sched.get_last_lr()[0], samples=bs)
            if val_time:
                hb.val(val_score)          # tracks best
    # exiting normally -> status "completed"; exception -> "failed" + traceback; Ctrl-C -> "interrupted"

Writes are atomic (tmp + os.replace) and throttled (every N steps or T seconds);
NaN/inf loss is written immediately with nan=True (and optionally raises).
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_runs_dir() -> Path:
    try:
        from src.config import get_paths

        return Path(get_paths().runs)
    except Exception:
        return Path(os.environ.get("RUNS_DIR") or "runs")


def atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    for attempt in range(20):  # Windows: replace fails briefly while a reader has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    for _ in range(5):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
    return None


class Heartbeat:
    def __init__(
        self,
        run_id: str,
        runs_dir: str | Path | None = None,
        every_steps: int = 50,
        every_seconds: float = 30.0,
        total_steps: int | None = None,
        metric_name: str = "val",
        higher_is_better: bool = True,
        nan_action: str = "flag",  # "flag" | "raise"
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.dir = Path(runs_dir or default_runs_dir()) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "heartbeat.json"
        self.every_steps = max(1, every_steps)
        self.every_seconds = every_seconds
        self.total_steps = total_steps
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.nan_action = nan_action
        self.t0 = time.time()
        self._last_write = 0.0
        self._last_write_step = -1
        self._rate_t, self._rate_samples, self._rate_step = self.t0, 0, 0
        self._samples_since = 0
        self.state: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": _now_iso(),
            "step": 0,
            "total_steps": total_steps,
            "epoch": None,
            "loss": None,
            "loss_min": None,
            "val_metric": None,
            "best_metric": None,
            "best_step": None,
            "metric_name": metric_name,
            "higher_is_better": higher_is_better,
            "lr": None,
            "samples_per_sec": None,
            "steps_per_sec": None,
            "eta_seconds": None,
            "nan": False,
            "elapsed_seconds": 0.0,
            "timestamp": _now_iso(),
            "unix_time": time.time(),
            "meta": meta or {},
        }
        atomic_write_json(
            self.dir / "run.json",
            {"run_id": run_id, "pid": os.getpid(), "argv": sys.argv, "started_at": self.state["started_at"],
             "host": self.state["host"], "meta": meta or {}},
        )
        self._write()

    # ------------------------------------------------------------------ API
    def step(self, step: int, loss: float | None = None, epoch: float | None = None,
             lr: float | None = None, samples: int = 0, force: bool = False, **extra: Any) -> None:
        self.state["step"] = int(step)
        if epoch is not None:
            self.state["epoch"] = epoch
        if lr is not None:
            self.state["lr"] = float(lr)
        self._samples_since += samples
        if extra:
            self.state.setdefault("extra", {}).update(extra)
        if loss is not None:
            loss = float(loss)
            self.state["loss"] = loss if math.isfinite(loss) else str(loss)
            if not math.isfinite(loss):
                self.state["nan"] = True
                self.state["nan_step"] = int(step)
                self._write()
                if self.nan_action == "raise":
                    raise FloatingPointError(f"non-finite loss {loss} at step {step}")
                return
            lm = self.state["loss_min"]
            self.state["loss_min"] = loss if lm is None else min(lm, loss)
        now = time.time()
        if force or step - self._last_write_step >= self.every_steps or now - self._last_write >= self.every_seconds:
            self._update_rates(now)
            self._write()

    def val(self, metric: float, step: int | None = None, **extra: Any) -> bool:
        """Record a validation score; returns True if it is a new best."""
        metric = float(metric)
        self.state["val_metric"] = metric
        best = self.state["best_metric"]
        improved = best is None or (metric > best if self.higher_is_better else metric < best)
        if improved and math.isfinite(metric):
            self.state["best_metric"] = metric
            self.state["best_step"] = self.state["step"] if step is None else step
        if extra:
            self.state.setdefault("extra", {}).update(extra)
        self._write()
        return improved

    def reset_best(self) -> None:
        """Start a new best-tracking window (e.g. next CV fold); the last value is kept in best_history."""
        if self.state["best_metric"] is not None:
            self.state.setdefault("best_history", []).append(self.state["best_metric"])
        self.state["best_metric"] = self.state["best_step"] = None

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.state["status"] = status
        self.state["finished_at"] = _now_iso()
        if error:
            self.state["error"] = error[-4000:]
        self._update_rates(time.time())
        self._write()

    # ------------------------------------------------------------ internals
    def _update_rates(self, now: float) -> None:
        dt = now - self._rate_t
        if dt <= 0:
            return
        dsteps = self.state["step"] - self._rate_step
        if dsteps > 0:
            sps = dsteps / dt
            self.state["steps_per_sec"] = round(sps, 4)
            if self._samples_since:
                self.state["samples_per_sec"] = round(self._samples_since / dt, 2)
            if self.total_steps:
                self.state["eta_seconds"] = round(max(0, self.total_steps - self.state["step"]) / sps, 1)
        self._rate_t, self._rate_step, self._samples_since = now, self.state["step"], 0

    def _write(self) -> None:
        now = time.time()
        self.state["elapsed_seconds"] = round(now - self.t0, 1)
        self.state["timestamp"] = _now_iso()
        self.state["unix_time"] = now
        atomic_write_json(self.path, self.state)
        self._last_write = now
        self._last_write_step = self.state["step"]

    def __enter__(self) -> Heartbeat:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.finish("completed")
        elif issubclass(exc_type, KeyboardInterrupt):
            self.finish("interrupted")
        else:
            self.finish("failed", "".join(traceback.format_exception(exc_type, exc, tb)))
        return False  # never swallow exceptions
