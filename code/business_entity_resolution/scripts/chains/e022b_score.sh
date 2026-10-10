#!/usr/bin/env bash
# E022 resumed from scoring (26 Sep 19:15): batch 32, because batch 64 spilled 4.3 GB of GPU memory into system RAM on
# the longer prompts (shards are length-sorted) and slowed scoring ~5x. Finished shards are skipped.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
for split in train test; do
  rm -f runs/E022-score-$split/exit.json
  $PY -m monitor.launch --run E022-score-$split --quiet -- $PY -m src.er_llmrank score --data runs/E021-llm/data10 \
      --lora runs/E022-llm/lora --split $split --out runs/E022-llm --batch 32
  ok E022-score-$split || { echo "CHAIN STOP: scoring $split failed"; exit 1; }
  echo "E022 score $split ok $(date +%H:%M)"
done
$PY -m src.er_llmrank to-ce --llm runs/E022-llm --out runs/E022-llmce || { echo "CHAIN STOP: to-ce failed"; exit 1; }
$PY -m monitor.launch --run E022-dev --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce --fit-folds 0 --tag E022 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --frames runs/frames/E022
ok E022-dev || { echo "CHAIN STOP: E022 stage 2 failed"; exit 1; }
mkdir -p runs/E015/E022_dev && cp runs/E015/stage2.json runs/E015/E022_dev/
echo "E022 CHAIN DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
