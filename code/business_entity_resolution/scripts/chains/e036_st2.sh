#!/usr/bin/env bash
# E036: second self-training round with a better teacher. Continue the round-1 self-trained e5-large CE (E030) on
# confident pseudo-labels (stage-2 >= 0.97 / <= 0.03) from E034 (owner-model features, best LB so far), France + US +
# India; re-score France candidate rows and US/India CE-uncertain rows. Stage 2 = the current final's inputs with
# runs/E030-ce swapped for runs/E036-ce. Train side is unchanged (E036-ce/train_ce is a copy of E030-ce's), so the
# final's train frame is reused and only the test frame is rebuilt. Same candidate filter -> identical candidates.
# Test-side only: dev cannot measure it; guards = validators, identical candidates, France drift guard, French
# uncertainty, US/India stability. Shipping it unprobed is a human decision.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
echo "E036 train start $(date +%H:%M)"
# France-only (27 Sep 16:57): a 3-country mix gave France only ~15% of the pseudo-labels; US/India self-training
# measured no gain (E031), so the whole budget goes to France (E018 used 600k French pseudo-labels on e5-base).
$PY -m monitor.launch --run E036f-selftrain --quiet -- $PY -m src.er_selftrain train \
    --probs runs/E015/sub_E034_ow04/test_probs_stage2.parquet --country France --base runs/E030-ce-st/model \
    --out runs/E036f-ce-st/model --n-pos 200000 --n-neg 250000 --batch 32 --lr 1e-5
ok E036f-selftrain || { echo "CHAIN STOP: E036 self-train failed"; exit 1; }
echo "E036 train ok $(date +%H:%M)"
$PY -m monitor.launch --run E036f-score-fr --quiet -- $PY -m src.er_selftrain score --model runs/E036f-ce-st/model \
    --country France --only-scored --base-ce runs/E030-ce --out runs/E036-ce --batch 256
ok E036f-score-fr || { echo "CHAIN STOP: E036 France scoring failed"; exit 1; }
echo "E036 scoring ok $(date +%H:%M)"
until grep -q "E035 DONE\|CHAIN STOP" runs/E035-chain.log 2>/dev/null; do sleep 30; done
until [ "$(stage2_running)" = 0 ]; do sleep 30; done
if grep -q "GATE PASS" runs/E035-chain.log; then BASE=E035_ce03; CE="runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E036-ce runs/OW04p runs/OW04m runs/CE03";
else BASE=E034_ow04; CE="runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E036-ce runs/OW04p runs/OW04m"; fi
echo "E036 stage 2 on top of $BASE $(date +%H:%M)"
mkdir -p runs/frames/E036_st2 && cp runs/frames/$BASE/train_frame.parquet runs/frames/E036_st2/
$PY -c "import json; k=json.load(open('runs/frames/$BASE/frames.json')); k['ce_dir']=[d.replace('runs/E030-ce','runs/E036-ce') for d in k['ce_dir']]; json.dump(k, open('runs/frames/E036_st2/frames.json','w'))"
$PY -m monitor.launch --run E036-final --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir $CE --fit-folds 0 --tag E036_st2 --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 \
    --comp-keep 0.78 --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 --unseen-threshold 0.85 \
    --frames runs/frames/E036_st2 --out submissions/sub_E036_st2
ok E036-final || { echo "CHAIN STOP: E036 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E036_st2 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E036_st2/
for v in sub_E036_st2 sub_E036_st2_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E036_st2/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
cmp -s submissions/sub_E033_e030_c383_unseen/candidate_pairs.tsv submissions/sub_E036_st2_unseen/candidate_pairs.tsv && echo "candidates byte-identical to E033" || echo "WARNING: candidates differ from E033"
$PY scripts/france_drift.py --base runs/E015/sub_$BASE --new runs/E015/sub_E036_st2
echo "E036 DONE $(date +%H:%M): dev $($PY -c "import json; r=json.load(open('runs/E015/sub_E036_st2/stage2.json')); print(round(r['dev_f05'],5), r['dev_by_country'])") (base $BASE)"
