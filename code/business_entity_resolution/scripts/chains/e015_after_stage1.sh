#!/usr/bin/env bash
# E015 chain: wait for stage 1, then CE (E008) scores for fold 0 + test, then stage 2 + stage 3 with outputs.
# Stops at the first failure (non-zero exit.json), so nothing downstream runs on partial inputs.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
until [ -f runs/E015-stage1/exit.json ]; do sleep 30; done
ok E015-stage1 || { echo "CHAIN STOP: stage 1 failed"; exit 1; }
echo "stage 1 ok $(date +%H:%M)"
$PY -m monitor.launch --run E015-ce-score-train --quiet -- $PY -m src.er_crossenc score --model runs/E008-ce/model \
    --chunks runs/E015/train_chunks --split train --out runs/E015-ce/train_ce --only-folds 0
ok E015-ce-score-train || { echo "CHAIN STOP: CE train scoring failed"; exit 1; }
echo "CE train ok $(date +%H:%M)"
$PY -m monitor.launch --run E015-ce-score-test --quiet -- $PY -m src.er_crossenc score --model runs/E008-ce/model \
    --chunks runs/E015/test_chunks --split test --out runs/E015-ce/test_ce
ok E015-ce-score-test || { echo "CHAIN STOP: CE test scoring failed"; exit 1; }
echo "CE test ok $(date +%H:%M)"
$PY -m monitor.launch --run E015-stage2 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 \
    --views name addr both both_ft --ce-dir runs/E015-ce --fit-folds 0 --stage3 --tag E015 --out submissions/sub_E015
ok E015-stage2 || { echo "CHAIN STOP: stage 2 failed"; exit 1; }
echo "CHAIN DONE $(date +%H:%M)"
