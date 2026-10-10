"""Verify the environment before Day 1: OS, CUDA/GPU (sm_120), bf16, matmul, libraries, .env keys.

    python scripts/check_env.py            # full check (GPU tests if available)
    python scripts/check_env.py --quick    # skip GBDT GPU training tests

Exit code 0 = ready; 1 = something required is broken. Never prints secret values.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, WARN, FAIL = "[ OK ]", "[WARN]", "[FAIL]"
problems: list[str] = []


def line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")
    if tag == FAIL:
        problems.append(msg)


def check_os() -> None:
    uname = platform.uname()
    wsl = "microsoft" in uname.release.lower()
    kind = "WSL2" if wsl else uname.system
    if uname.system == "Windows":  # release says "10" on Windows 11 too; build >= 22000 means 11
        build = int(uname.version.split(".")[-1]) if uname.version.split(".")[-1].isdigit() else 0
        kind, uname = f"Windows {'11' if build >= 22000 else '10'} (build {build})", uname._replace(release="")
    line(OK, f"OS: {kind} {uname.release} | Python {platform.python_version()} ({sys.executable})")
    if uname.system == "Windows":
        line(WARN, "native Windows: bitsandbytes/flash-attn/torch.compile(inductor) are limited — see docs/HARDWARE.md (WSL2)")
        try:
            import subprocess

            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-MpComputerStatus).SmartAppControlState"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out == "On":
                line(WARN, "Smart App Control is ON — it can block unsigned DLLs (torch.dll, pandas). See docs/HARDWARE.md")
        except Exception:
            pass
    if sys.version_info < (3, 10):
        line(FAIL, "Python >= 3.10 required")


def check_resources() -> None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        line(OK, f"CPU: {psutil.cpu_count(logical=False)}C/{psutil.cpu_count()}T | RAM {vm.total / 2**30:.0f} GB "
                 f"({vm.available / 2**30:.0f} GB free)")
    except ImportError:
        line(WARN, "psutil missing")
    free = shutil.disk_usage(ROOT).free / 2**30
    line(OK if free > 100 else WARN, f"disk free at repo: {free:.0f} GB")


def check_torch() -> None:
    try:
        import torch
    except Exception as e:  # ImportError or OSError (DLL blocked)
        line(FAIL, f"torch import failed: {type(e).__name__}: {str(e)[:200]}")
        return
    line(OK, f"torch {torch.__version__} | CUDA build {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
    if not torch.cuda.is_available():
        line(FAIL, "torch.cuda.is_available() is False (CPU-only build or driver issue)")
        return
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    archs = torch.cuda.get_arch_list()
    line(OK, f"GPU: {name} | capability sm_{cap[0]}{cap[1]} | {vram:.1f} GB VRAM")
    if cap >= (12, 0) and not any(a.startswith("sm_12") for a in archs):
        line(FAIL, f"torch build lacks sm_120 kernels (arch list {archs}); install cu128+ wheels")
    else:
        line(OK, f"arch list: {' '.join(archs)}")
    cv = torch.version.cuda or "0"
    if tuple(int(x) for x in cv.split(".")[:2]) < (12, 8):
        line(FAIL, f"CUDA build {cv} < 12.8 — Blackwell needs cu128+")
    line(OK if torch.cuda.is_bf16_supported() else FAIL, f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    # correctness + throughput
    try:
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        ref = (a.double() @ b.double()).float()
        err = ((a @ b) - ref).abs().max().item()
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            x, y = a.to(dt), b.to(dt)
            for _ in range(3):
                x @ y
            torch.cuda.synchronize()
            t = time.perf_counter()
            n = 20
            for _ in range(n):
                x @ y
            torch.cuda.synchronize()
            tflops = 2 * 4096**3 * n / (time.perf_counter() - t) / 1e12
            line(OK, f"matmul {str(dt).split('.')[-1]:8} {tflops:6.1f} TFLOPS")
        line(OK if err < 1.0 else FAIL, f"fp32 matmul max abs err vs fp64: {err:.3g} (TF32 {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'})")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            net = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 1)).cuda()
            loss = net(torch.randn(64, 512, device="cuda")).pow(2).mean()
        loss.backward()
        line(OK, "bf16 autocast forward/backward")
        print(f"       peak VRAM used by checks: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    except Exception as e:
        line(FAIL, f"GPU compute test failed: {type(e).__name__}: {e}")


LIBS = ["numpy", "pandas", "pyarrow", "sklearn", "scipy", "lightgbm", "xgboost", "catboost", "optuna",
        "transformers", "tokenizers", "accelerate", "peft", "timm", "sentence_transformers", "safetensors",
        "torchvision", "PIL", "yaml", "dotenv", "rich", "psutil", "pynvml", "requests", "tqdm", "pytest"]
OPTIONAL = {"bitsandbytes", "flash_attn", "open_clip", "cv2"}


def check_libs() -> None:
    for name in LIBS + sorted(OPTIONAL):
        try:
            m = importlib.import_module(name)
            line(OK, f"{name:22} {getattr(m, '__version__', '')}")
        except Exception as e:
            tag = WARN if name in OPTIONAL else FAIL
            line(tag, f"{name:22} not importable ({type(e).__name__}: {str(e)[:90]})")


def check_gbdt_gpu() -> None:
    import numpy as np

    X = np.random.rand(2000, 20).astype("float32")
    y = (X[:, 0] + np.random.rand(2000) * 0.1).astype("float32")
    try:
        import xgboost as xgb

        xgb.XGBRegressor(n_estimators=20, device="cuda", tree_method="hist").fit(X, y)
        line(OK, "xgboost GPU (device='cuda') trains")
    except Exception as e:
        line(WARN, f"xgboost GPU failed: {str(e)[:120]}")
    try:
        from catboost import CatBoostRegressor

        CatBoostRegressor(iterations=20, task_type="GPU", verbose=0).fit(X, y)
        line(OK, "catboost GPU (task_type='GPU') trains")
    except Exception as e:
        line(WARN, f"catboost GPU failed: {str(e)[:120]}")
    try:
        import lightgbm as lgb

        lgb.LGBMRegressor(n_estimators=20, device_type="gpu", verbose=-1).fit(X, y)
        line(OK, "lightgbm GPU (OpenCL device_type='gpu') trains")
    except Exception as e:
        line(WARN, f"lightgbm GPU unavailable (CPU is fine & fast on 32 threads): {str(e)[:80]}")


def check_nvml_env() -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        line(OK, f"NVML: driver {pynvml.nvmlSystemGetDriverVersion()} | temp "
                 f"{pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)}°C")
    except Exception as e:
        line(WARN, f"NVML unavailable (monitor will skip GPU stats): {e}")
    env = ROOT / ".env"
    if not env.exists():
        line(WARN, ".env missing — copy .env.example to .env (Discord alerts disabled until then)")
        return
    try:
        from dotenv import dotenv_values

        vals = dotenv_values(env)
    except ImportError:
        vals = {}
    for key in ("DISCORD_WEBHOOK_URL", "DATA_DIR"):
        line(OK if vals.get(key) else WARN, f".env {key}: {'set' if vals.get(key) else 'NOT set'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONUTF8", "1")
    print("== system")
    check_os()
    check_resources()
    print("== torch / GPU")
    check_torch()
    print("== libraries")
    check_libs()
    if not a.quick:
        print("== GBDT GPU")
        check_gbdt_gpu()
    print("== monitor / .env")
    check_nvml_env()
    print("\nRESULT:", "READY" if not problems else f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
