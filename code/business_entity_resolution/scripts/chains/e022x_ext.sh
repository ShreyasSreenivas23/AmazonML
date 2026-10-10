#!/usr/bin/env bash
# E022x: score the wider uncertain band (E016 CE in (0.02,0.1] or [0.9,0.98)) with the same E022 LoRA, overnight,
# then map core + extended scores into one stage-2 feature dir (runs/E022-llmce-full) for the final build.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
until ok E022-score-test; do sleep 60; done                         # GPU free once the core test band is scored
for split in train test; do
  $PY -m monitor.launch --run E022x-score-$split --quiet -- $PY -m src.er_llmrank score --data runs/E021-llm/data_ext \
      --lora runs/E022-llm/lora --split $split --out runs/E022x-llm --batch 32
  ok E022x-score-$split || { echo "CHAIN STOP: extended scoring $split failed"; exit 1; }
  echo "E022x score $split ok $(date +%H:%M)"
done
mkdir -p runs/E022-llm-full
for split in train test; do
  mkdir -p runs/E022-llm-full/llm_$split
  for f in runs/E022-llm/llm_$split/*.parquet; do cp "$f" runs/E022-llm-full/llm_$split/core_$(basename "$f"); done
  for f in runs/E022x-llm/llm_$split/*.parquet; do cp "$f" runs/E022-llm-full/llm_$split/ext_$(basename "$f"); done
done
$PY -m src.er_llmrank to-ce --llm runs/E022-llm-full --out runs/E022-llmce-full || { echo "CHAIN STOP: to-ce failed"; exit 1; }
echo "E022x CHAIN DONE $(date +%H:%M)"
