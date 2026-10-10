#!/usr/bin/env bash
# E025 = E024 (norm2) + density-matched competition on train (--comp-keep 0.78 -> ~2.67 S1s per record, like test).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E024-dev/exit.json ]; do sleep 30; done
$PY -m monitor.launch --run E025-dev --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce --fit-folds 0 --tag E025 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 --frames runs/frames/E025
grep -q '"returncode": 0' runs/E025-dev/exit.json || { echo "CHAIN STOP: E025 failed"; exit 1; }
mkdir -p runs/E015/E025_dev && cp runs/E015/stage2.json runs/E015/E025_dev/
echo "E025 DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
