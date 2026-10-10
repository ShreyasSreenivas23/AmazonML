#!/usr/bin/env bash
# E016: train the e5-base cross-encoder once E015 stage 1 has exited cleanly (RAM is free then).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E015-stage1/exit.json ]; do sleep 30; done
grep -q '"returncode": 0' runs/E015-stage1/exit.json || { echo "CHAIN STOP: stage 1 failed"; exit 1; }
rm -f runs/E016-ce-base-train/exit.json
$PY -m monitor.launch --run E016-ce-base-train --quiet -- $PY -m src.er_crossenc train --chunks runs/E015/train_chunks \
    --out runs/E016-ce-base/model --n 3000000 --exclude-folds 0 --model-name intfloat/multilingual-e5-base \
    --batch 64 --lr 2e-5 --ckpt-every 2000
grep -q '"returncode": 0' runs/E016-ce-base-train/exit.json && echo "E016 TRAIN DONE $(date +%H:%M)" || echo "CHAIN STOP: E016 train failed"
