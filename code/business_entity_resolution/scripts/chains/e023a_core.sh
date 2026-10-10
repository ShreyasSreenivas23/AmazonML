#!/usr/bin/env bash
# E023a (core reranker band): no self-training (plain E016 CE on test), 4.71/S1 candidate filter, norm2, test-like
# density, unseen-country variant from the same run. Waits for E022's CPU stage-2 run (RAM: one stage 2 at a time).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E022-dev/exit.json ] || grep -q "CHAIN STOP" runs/E022b-chain.log 2>/dev/null; do sleep 60; done
[ -d runs/E022-llmce/test_ce ] && [ "$(ls runs/E022-llmce/test_ce | wc -l)" = 9 ] || { echo "CHAIN STOP: runs/E022-llmce incomplete"; exit 1; }
$PY -m monitor.launch --run E023a-core --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E016-ce runs/E022-llmce --fit-folds 0 --tag E023a_core \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E023a_core --out submissions/sub_E023a_core
grep -q '"returncode": 0' runs/E023a-core/exit.json || { echo "CHAIN STOP: E023a-core stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E023a_core && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E023a_core/
for v in sub_E023a_core sub_E023a_core_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E023a_core/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
echo "E023a-core DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/stage2.json')); p=json.load(open('runs/E015/predict.json')); print('dev', r['dev_f05'], r['dev_by_country'], '| test pairs', p['test_pairs'], 'cands/S1', round(p['test_pairs']/1732544,2))")"
