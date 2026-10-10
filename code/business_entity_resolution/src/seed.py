"""Reproducibility: seed every RNG we use."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python, numpy, torch (if installed).

    deterministic=True trades speed for bitwise reproducibility (cuDNN deterministic,
    no benchmark). Leave False for normal training; use True when debugging a CV diff.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Pass to DataLoader(worker_init_fn=...) so each worker has a distinct, reproducible seed."""
    import torch  # inside a worker, torch.initial_seed() is already base_seed + worker_id

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
