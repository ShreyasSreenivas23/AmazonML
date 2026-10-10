#!/usr/bin/env bash
# E016 stage 2: E015 candidates + both cross-encoders (E008 small, E016 e5-base) as features; no stage 3 (no gain in E015).
# Reads runs/E015 chunks, so it overwrites runs/E015/stage2.json etc. (E015's copies are in runs/E015/sub08_E015/).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E016-ce-score-test/exit.json ]; do sleep 30; done
grep -q '"returncode": 0' runs/E016-ce-score-test/exit.json || { echo "CHAIN STOP: E016 scoring failed"; exit 1; }
$PY -m monitor.launch --run E016-stage2 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 \
    --views name addr both both_ft --ce-dir runs/E015-ce runs/E016-ce --fit-folds 0 --tag E016 --out submissions/sub_E016
grep -q '"returncode": 0' runs/E016-stage2/exit.json && echo "E016 CHAIN DONE $(date +%H:%M)" || echo "CHAIN STOP: E016 stage 2 failed"
