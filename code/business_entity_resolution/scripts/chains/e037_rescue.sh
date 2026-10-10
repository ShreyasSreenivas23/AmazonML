#!/usr/bin/env bash
# E037: E036 + unseen-country candidate rescue. For S1 countries absent from train (France), also keep pairs that the
# France-adapted CE (runs/E036-ce: round-2 self-trained e5-large, French rows) scores >= 0.5; the filter CE (E016) never
# saw French text and dropped ~0.009 likely French matches per S1. Same frames as E036 (prune happens after the
# frames), so this is a refit from cache. Train rows (US/India) are unchanged -> the stage-2 model and dev are identical.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
until grep -q "E036 DONE\|CHAIN STOP" runs/E036-chain.log; do sleep 30; done
grep -q "E036 DONE" runs/E036-chain.log || { echo "CHAIN STOP: E036 not done"; exit 1; }
until [ "$(stage2_running)" = 0 ]; do sleep 30; done
if grep -q "on top of E035_ce03" runs/E036-chain.log; then CE="runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E036-ce runs/OW04p runs/OW04m runs/CE03";
else CE="runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E036-ce runs/OW04p runs/OW04m"; fi
echo "E037 start $(date +%H:%M)"
$PY -m monitor.launch --run E037-rescue --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir $CE --fit-folds 0 --tag E037_rescue --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 \
    --comp-keep 0.78 --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 \
    --prune-unseen-ce-dir runs/E036-ce --prune-unseen-ce 0.5 --unseen-threshold 0.85 \
    --frames runs/frames/E036_st2 --out submissions/sub_E037_rescue
grep -q '"returncode": 0' runs/E037-rescue/exit.json || { echo "CHAIN STOP: E037 failed"; exit 1; }
mkdir -p runs/E015/sub_E037_rescue && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E037_rescue/
grep "rescue adds\|candidate set" runs/E037-rescue/train.log | tail -3
for v in sub_E037_rescue sub_E037_rescue_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E037_rescue/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
$PY scripts/cand_recall.py --tag E037_rescue --sub submissions/sub_E037_rescue_unseen --frames runs/frames/E036_st2 --prune-eps 0.5 --prune-ce 0.05
echo "E037 DONE $(date +%H:%M): dev $($PY -c "import json; r=json.load(open('runs/E015/sub_E037_rescue/stage2.json')); print(round(r['dev_f05'],5))")"
