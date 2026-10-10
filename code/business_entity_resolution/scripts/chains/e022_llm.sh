#!/usr/bin/env bash
# E022: Qwen3-4B LoRA reranker (competitor-aware prompts, E021 data) on the local GPU, then stage 2 with its score.
# GPU after E020's self-training/scoring. Smoke test first so a crash costs minutes, not hours.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
# GPU free since E020 scoring finished (15:54). Data: (0.1, 0.9) CE band, 40k train prompts (runs/E021-llm/data10).
$PY -m monitor.launch --run E022-smoke --quiet -- $PY -m src.er_llmrank train --data runs/E021-llm/data10 \
    --out runs/E022-smoke/lora --max-rows 200 --batch 8
ok E022-smoke || { echo "CHAIN STOP: smoke test failed"; exit 1; }
echo "E022 smoke ok $(date +%H:%M)"
$PY -m monitor.launch --run E022-train --quiet -- $PY -m src.er_llmrank train --data runs/E021-llm/data10 \
    --out runs/E022-llm/lora --batch 16 --ckpt-every 500
ok E022-train || { echo "CHAIN STOP: LoRA training failed"; exit 1; }
echo "E022 train ok $(date +%H:%M)"
for split in train test; do
  $PY -m monitor.launch --run E022-score-$split --quiet -- $PY -m src.er_llmrank score --data runs/E021-llm/data10 \
      --lora runs/E022-llm/lora --split $split --out runs/E022-llm --batch 64
  ok E022-score-$split || { echo "CHAIN STOP: scoring $split failed"; exit 1; }
  echo "E022 score $split ok $(date +%H:%M)"
done
$PY -m src.er_llmrank to-ce --llm runs/E022-llm --out runs/E022-llmce || { echo "CHAIN STOP: to-ce failed"; exit 1; }
until [ -f runs/E020-test/exit.json ]; do sleep 30; done     # RAM: never overlap E020's final stage 2
$PY -m monitor.launch --run E022-dev --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce --fit-folds 0 --tag E022 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --frames runs/frames/E022
ok E022-dev || { echo "CHAIN STOP: E022 stage 2 failed"; exit 1; }
mkdir -p runs/E015/E022_dev && cp runs/E015/stage2.json runs/E015/E022_dev/
echo "E022 CHAIN DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
