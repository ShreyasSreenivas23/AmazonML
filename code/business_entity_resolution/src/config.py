"""Tiny YAML config system with inheritance and dotted CLI overrides.

    cfg = load_config("configs/example.yaml", ["train.lr=3e-4", "seed=7"])
    cfg.train.lr        # attribute access
    cfg["train"]["lr"]  # dict access still works
    save_config(cfg, run_dir / "config.yaml")

A config may declare `defaults: base.yaml` (path relative to itself); the child is
deep-merged over the parent. Paths to data/runs come from .env (see get_paths),
never from committed configs, so the repo runs anywhere (local GPU, WSL, cloud).
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        """Attribute access maps to dict keys (cfg.model.lr)."""
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        """Attribute assignment writes the dict key."""
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        """Wrap nested dicts so every level supports attribute access."""
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        """Return a plain (unwrapped) dict copy of the config."""
        def unwrap(o: Any) -> Any:
            """Convert wrapped config objects back to plain dicts recursively."""
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o

        return unwrap(self)


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge dict b into a copy of a (b wins on conflicts)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict (empty dict for an empty file)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("defaults", None)
    if parent:
        data = _deep_merge(_load_yaml((path.parent / parent).resolve()), data)
    return data


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    """Set a nested key given as 'a.b.c' inside dict d, creating levels as needed."""
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load YAML config, apply `a.b.c=value` overrides (values parsed as YAML)."""
    data = _load_yaml(Path(path).resolve())
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(data, k.strip(), yaml.safe_load(v))
    return Config.wrap(data)


def save_config(cfg: dict, path: str | Path) -> None:
    """Write the resolved config to a YAML file next to the run artefacts."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_env() -> None:
    """Load .env from repo root if python-dotenv is available (never overrides real env)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def get_paths() -> Config:
    """Resolve DATA_DIR / RUNS_DIR / CKPT_DIR from env (.env), defaulting to repo-local dirs."""
    load_env()

    def resolve(var: str, default: str) -> Path:
        """Load base.yaml, merge an experiment config and CLI overrides, and return the wrapped result."""
        p = Path(os.environ.get(var) or default)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    return Config(data=resolve("DATA_DIR", "data"), runs=resolve("RUNS_DIR", "runs"),
                  ckpt=resolve("CKPT_DIR", "checkpoints"))
