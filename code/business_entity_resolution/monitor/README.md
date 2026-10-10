# monitor â€” training heartbeat + watchdog + Discord alerts

Know at all times what is training, whether it is healthy, and when it's done â€” even when asleep.

```
training script â”€â”€Heartbeatâ”€â”€â–¶ runs/<run>/heartbeat.json â—€â”€â”€readsâ”€â”€ monitor.watch â”€â”€â–¶ rich dashboard
      â–²                              runs/<run>/train.log (via monitor.launch)       â”€â”€â–¶ runs/<run>/watch.csv
      â””â”€ monitor.launch (log capture, exit code â†’ exit.json)                          â”€â”€â–¶ Discord (@here on failure) / ntfy
```

## 1. Instrument the training loop
```python
import os
from monitor import Heartbeat

with Heartbeat(os.environ.get("RUN_ID", "E007-x"), total_steps=epochs * len(dl),
               metric_name="smape", higher_is_better=False) as hb:
    for epoch in range(epochs):
        for step, batch in enumerate(dl, start=...):
            ...
            hb.step(global_step, loss=loss.item(), epoch=epoch, lr=sched.get_last_lr()[0], samples=bs)
        hb.val(val_score)            # records best; returns True if improved
```
- Writes every `every_steps` (50) or `every_seconds` (30) â€” cheap; atomic (tmp + rename).
- NaN/inf loss â†’ written **immediately** with `nan=True` (`nan_action="raise"` to stop training).
- Context manager marks `completed`, `failed` (with traceback), or `interrupted` (Ctrl-C).
- Full template: `src/train_template.py`.

## 2. Launch (recommended) and watch
```bash
# terminal 1: runs the command, tees output to runs/<run>/train.log, records exit code
python -m monitor.launch --run E007-deberta -- python -m src.train_template --cfg configs/E007.yaml
# terminal 2
python -m monitor.watch --run E007-deberta
```
Without `monitor.launch` the watchdog still works (PID from heartbeat), but can't read tracebacks
unless you pass `--log <file>`.

## What the watchdog checks
| Check | Default | Level |
|---|---|---|
| Run failed / interrupted (heartbeat status) â€” OOM detected from traceback | â€” | ðŸ”´ critical, @here |
| Process exited with non-zero code (`exit.json`) / PID vanished (hard kill, segfault) | â€” | ðŸ”´ critical, @here |
| Heartbeat stale (hang, deadlocked dataloader) | `--stale-min 10` | ðŸ”´ critical, @here |
| No heartbeat file after startup | `--stale-min` | ðŸ”´ critical |
| Traceback / OOM / CUDA error text in log while still "running" | â€” | ðŸ”´ critical |
| NaN/inf loss | immediate | ðŸ”´ critical |
| Loss diverging (loss > 3Ã— min, after step 100) | `--diverge-factor 3` | ðŸŸ¡ warning |
| GPU util < 5% for too long while running | `--gpu-idle-min 10` | ðŸŸ¡ warning |
| GPU temp | `--temp-c 85` | ðŸŸ¡ warning |
| Disk free / RAM | `--disk-min-gb 20`, 95% | ðŸŸ¡ warning |
| Periodic status embed | `--summary-min 20` | ðŸ”µ info |
| Run completed (with best score) | â€” | ðŸŸ¢ success |

Alerts are de-duplicated per condition (re-sent after 30 min if it persists). The watchdog exits after
the run ends (`--keep` to stay). `--once` prints a JSON snapshot â€” the cheap way to check a run
without dumping logs. `--no-dashboard` prints JSON lines (for nohup/tmux logs).

## Notifications
Put in `.env` (gitignored; the URL is never printed or logged):
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
NTFY_TOPIC_URL=https://ntfy.sh/<long-random-topic>   # optional fallback (phone push)
OWNER=yourname                                        # optional footer tag
```
ntfy is used when Discord fails/unset, and additionally for every critical alert.
```bash
python -m monitor.watch --test-notify     # sends one info + one critical test message; exit 0 if delivered
```

## Prove it works
```bash
python -m monitor.demo            # 5 simulated runs, dry-run (no network): ok, OOM crash, NaN, hang, hard kill
python -m monitor.demo --notify   # same, but really posts to Discord/ntfy
python -m pytest tests/test_monitor.py
```
Expected: every scenario `PASS`. Demo runs live under `runs/_demo/` (gitignored).
