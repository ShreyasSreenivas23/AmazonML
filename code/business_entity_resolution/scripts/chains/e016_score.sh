#!/usr/bin/env bash
# E016: score E015 candidates with the e5-base cross-encoder (fold 0 + test). Stage 2 is launched separately.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
$PY -m monitor.launch --run E016-ce-score-train --quiet -- $PY -m src.er_crossenc score --model runs/E016-ce-base/model \
    --chunks runs/E015/train_chunks --split train --out runs/E016-ce/train_ce --only-folds 0
ok E016-ce-score-train || { echo "CHAIN STOP: E016 train scoring failed"; exit 1; }
echo "E016 CE train ok $(date +%H:%M)"
$PY -m monitor.launch --run E016-ce-score-test --quiet -- $PY -m src.er_crossenc score --model runs/E016-ce-base/model \
    --chunks runs/E015/test_chunks --split test --out runs/E016-ce/test_ce
ok E016-ce-score-test || { echo "CHAIN STOP: E016 test scoring failed"; exit 1; }
echo "E016 CE test ok $(date +%H:%M)"
