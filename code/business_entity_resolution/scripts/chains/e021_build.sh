#!/usr/bin/env bash
# E021 reranker prompts (CPU/RAM): run in the gap after E019's dev part and before E020's stage 2.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E019-dev/exit.json ]; do sleep 30; done
$PY -m monitor.launch --run E021-build --quiet -- $PY -m src.er_llmrank build --out runs/E021-llm/data
grep -q '"returncode": 0' runs/E021-build/exit.json && echo "E021 BUILD DONE $(date +%H:%M)" || echo "CHAIN STOP: E021 build failed"
