#!/usr/bin/env bash
# E029-final: max stack for the final submission. Stage 2 = E027 inputs + self-trained CE on test (runs/E020-ce, user
# decision 27 Sep 11:15) + E029 (E027 CE continued on francized train pairs; only unseen-country test rows re-scored).
# Train side is identical to E027's (E020-ce/train_ce = E016's, E029-ce/train_ce = E027's), so E027's train frame is
# reused and only the test frame is rebuilt. Same candidate filter -> candidate_pairs.tsv byte-identical to sub-12.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
until grep -q "E029 score ok\|CHAIN STOP" runs/E029-chain.log 2>/dev/null; do sleep 60; done
grep -q "E029 score ok" runs/E029-chain.log || { echo "CHAIN STOP: E029 scoring not ok"; exit 1; }
until [ -f runs/E027-stage2/exit.json ] && [ "$(stage2_running)" = 0 ]; do sleep 60; done
[ "$(ls runs/E029-ce/test_ce/*.parquet | wc -l)" = 9 ] || { echo "CHAIN STOP: E029-ce incomplete"; exit 1; }
mkdir -p runs/frames/E029_final && cp runs/frames/E027/train_frame.parquet runs/frames/E029_final/
$PY -c "import json; k=json.load(open('runs/frames/E027/frames.json')); k['ce_dir']=['runs/E015-ce','runs/E020-ce','runs/E022-llmce-full','runs/E029-ce']; json.dump(k, open('runs/frames/E029_final/frames.json','w'))"
echo "E029-final stage 2 start $(date +%H:%M)"
$PY -m monitor.launch --run E029-final --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E029-ce --fit-folds 0 --tag E029_final \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E029_final --out submissions/sub_E029_final
grep -q '"returncode": 0' runs/E029-final/exit.json || { echo "CHAIN STOP: E029-final stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E029_final && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E029_final/
for v in sub_E029_final sub_E029_final_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E029_final/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
  cmp -s submissions/sub_E023b_full/candidate_pairs.tsv submissions/$v/candidate_pairs.tsv && echo "$v candidates identical to sub-12" || echo "WARNING: $v candidates differ from sub-12"
done
$PY scripts/cand_recall.py --tag E029_final --sub submissions/sub_E029_final_unseen
echo "E029-final DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/sub_E029_final/stage2.json')); p=json.load(open('runs/E015/sub_E029_final/predict.json')); print('dev', r['dev_f05'], r['dev_by_country'], '| mean matches', round(p['mean_matches'],3))")"
