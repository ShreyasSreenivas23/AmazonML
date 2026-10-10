#!/usr/bin/env bash
# E018-fr: continue the E016 cross-encoder on confident French test pseudo-labels, then re-score French test rows.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
$PY -m monitor.launch --run E018-selftrain --quiet -- $PY -m src.er_selftrain train \
    --probs runs/E015/sub_E016/test_probs_stage2.parquet --country France --base runs/E016-ce-base/model \
    --out runs/E018-ce-fr/model
ok E018-selftrain || { echo "CHAIN STOP: self-train failed"; exit 1; }
echo "E018 train ok $(date +%H:%M)"
$PY -m monitor.launch --run E018-score --quiet -- $PY -m src.er_selftrain score --model runs/E018-ce-fr/model \
    --country France --base-ce runs/E016-ce --out runs/E018-ce
ok E018-score || { echo "CHAIN STOP: France re-scoring failed"; exit 1; }
echo "E018 CHAIN DONE $(date +%H:%M)"
