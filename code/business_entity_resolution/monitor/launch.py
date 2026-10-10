"""Launch a training command with log capture + exit-code recording for the watchdog.

    python -m monitor.launch --run E007-deberta-f0 -- python -m src.train_template --cfg configs/example.yaml

- streams child stdout+stderr to the console AND runs/<run>/train.log (tracebacks/OOM land there)
- writes runs/<run>/launch.json (child pid, cmd) and runs/<run>/exit.json (return code) on exit
- exports RUN_ID=<run> so the training script can do Heartbeat(os.environ["RUN_ID"])
Works identically on Windows, WSL, and Linux. Exit code = child's exit code.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from monitor.heartbeat import atomic_write_json, default_runs_dir


def launch(run_id: str, cmd: list[str], runs_dir: str | Path | None = None, quiet: bool = False) -> int:
    d = Path(runs_dir or default_runs_dir()) / run_id
    d.mkdir(parents=True, exist_ok=True)
    if cmd and Path(cmd[0]).name.lower() in ("python", "python.exe", "python3", "py"):
        # On Windows, CreateProcess finds the *base* interpreter before the venv's; pin to ours.
        cmd = [sys.executable, *cmd[1:]]
    for stale in ("exit.json",):
        (d / stale).unlink(missing_ok=True)
    env = {**os.environ, "RUN_ID": run_id, "RUNS_DIR": str(d.parent), "PYTHONUNBUFFERED": "1",
           "PYTHONIOENCODING": "utf-8"}
    log = open(d / "train.log", "ab")
    header = f"\n===== launch {datetime.now(timezone.utc).isoformat()} :: {' '.join(cmd)} =====\n"
    log.write(header.encode())
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    atomic_write_json(d / "launch.json", {"run_id": run_id, "pid": proc.pid, "launcher_pid": os.getpid(),
                                          "cmd": cmd, "started_at": datetime.now(timezone.utc).isoformat()})
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, b""):
            log.write(line)
            log.flush()
            if not quiet:
                sys.stdout.write(line.decode("utf-8", "replace"))
                sys.stdout.flush()
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
    finally:
        log.close()
    atomic_write_json(d / "exit.json", {"returncode": rc, "finished_at": datetime.now(timezone.utc).isoformat()})
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m monitor.launch")
    ap.add_argument("--run", required=True, help="run id, e.g. E007-deberta-f0")
    ap.add_argument("--runs-dir")
    ap.add_argument("--quiet", action="store_true", help="log only, no console echo")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command ...>")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("missing command after --")
    return launch(a.run, cmd, a.runs_dir, a.quiet)


if __name__ == "__main__":
    sys.exit(main())
