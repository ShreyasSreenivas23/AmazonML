#!/usr/bin/env bash
# E023b (G0): E023a-core with the reranker on the WIDER band (runs/E022-llmce-full = core band + E022x extension).
# One full stage-2 run (dev + test) instead of dev-only then rebuild: if its dev >= E023a-core it is ready to submit,
# otherwise it is simply not used. Same candidate filter (stage-1 >= 0.2 OR E016 CE >= 0.01) => candidate_pairs.tsv must
# be byte-identical to E023a-core. Plain E016 CE on test (no self-training) unless organisers confirm it is allowed.
# Waits for the previous chat's chains: GPU scoring (E022x) and the last CPU stage 2 (E023a-st).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until grep -q "E022x CHAIN DONE\|CHAIN STOP" runs/E022x-chain.log 2>/dev/null; do sleep 60; done
grep -q "E022x CHAIN DONE" runs/E022x-chain.log || { echo "CHAIN STOP: E022x did not finish"; exit 1; }
until grep -q "E023a-st DONE\|CHAIN STOP" runs/E023a-st-chain.log 2>/dev/null; do sleep 60; done   # RAM: one stage 2
[ "$(ls runs/E022-llmce-full/test_ce | wc -l)" = 9 ] || { echo "CHAIN STOP: runs/E022-llmce-full incomplete"; exit 1; }
$PY -m monitor.launch --run E023b-full --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E016-ce runs/E022-llmce-full --fit-folds 0 --tag E023b_full \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E023b_full --out submissions/sub_E023b_full
grep -q '"returncode": 0' runs/E023b-full/exit.json || { echo "CHAIN STOP: E023b-full stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E023b_full && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E023b_full/
for v in sub_E023b_full sub_E023b_full_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E023b_full/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
cmp -s submissions/sub_E023a_core/candidate_pairs.tsv submissions/sub_E023b_full/candidate_pairs.tsv && echo "candidate_pairs identical to E023a-core" || echo "WARNING: candidate_pairs differ from E023a-core"
$PY scripts/cand_recall.py --tag E023b_full --sub submissions/sub_E023b_full --frames runs/frames/E023b_full
echo "E023b-full DONE $(date +%H:%M): full $($PY -c "import json; r=json.load(open('runs/E015/sub_E023b_full/stage2.json')); print(r['dev_f05'], r['dev_by_country'])") | core $($PY -c "import json; r=json.load(open('runs/E015/sub_E023a_core/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
