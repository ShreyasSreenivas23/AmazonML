#!/usr/bin/env bash
# E022 early dev check: as soon as fold-0 LLM scores exist, map them and run stage 2 dev-only on the CPU
# (the GPU keeps scoring test). Answers "does the reranker help?" hours earlier.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until grep -q '"returncode": 0' runs/E022-score-train/exit.json 2>/dev/null; do sleep 30; done
$PY -m src.er_llmrank to-ce --llm runs/E022-llm --out runs/E022-llmce-dev --splits train || { echo "CHAIN STOP: to-ce"; exit 1; }
mkdir -p runs/E022-llmce-dev/test_ce
$PY -m monitor.launch --run E022-earlydev --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-dev --fit-folds 0 --tag E022early \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort
grep -q '"returncode": 0' runs/E022-earlydev/exit.json || { echo "CHAIN STOP: early dev failed"; exit 1; }
mkdir -p runs/E015/E022_earlydev && cp runs/E015/stage2.json runs/E015/E022_earlydev/
echo "E022 EARLY DEV $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
