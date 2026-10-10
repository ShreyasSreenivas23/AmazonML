#!/usr/bin/env bash
# E017: E016 (both CEs) + full-population name competition features (off-the-shelf columns only: cos_both_ft
# competitors are mostly fold 1-4 S1s the bi-encoder trained on, a bias dev cannot reveal). Runs after E016 stage 2.
# Stage 2 reads runs/E015 chunks and overwrites runs/E015/stage2.json etc., so each run's outputs are copied aside.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E016-stage2/exit.json ]; do sleep 30; done
grep -q '"returncode": 0' runs/E016-stage2/exit.json || { echo "CHAIN STOP: E016 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E016 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E016/
$PY -m monitor.launch --run E017-stage2 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 \
    --views name addr both both_ft --ce-dir runs/E015-ce runs/E016-ce --fit-folds 0 --tag E017 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --out submissions/sub_E017
grep -q '"returncode": 0' runs/E017-stage2/exit.json || { echo "CHAIN STOP: E017 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E017 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E017/
echo "E017 CHAIN DONE $(date +%H:%M)"
