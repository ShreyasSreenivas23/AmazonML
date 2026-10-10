#!/usr/bin/env bash
# E030: E018/E020 self-training recipe on the large CE. Continue BASE (E029 francized e5-large, or E027 if E029 fails its
# check) on confident test pseudo-labels (stage-2 >= 0.97 / <= 0.03 from E023b-ST) for France + US + India, mixed with
# original train pairs; re-score France candidate rows and US/India uncertain candidate rows (only rows the base CE
# scored). Then stage 2 = E015-ce + E020-ce + E022-llmce-full + E030-ce (train frame = E027's; same candidates).
# Usage: BASE=E029 bash scripts/chains/e030_selftrain_large.sh   (BASE=E027 falls back to the plain large CE)
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
BASE=${BASE:-E029}
if [ "$BASE" = E029 ]; then BM=runs/E029-ce-fr/model; BCE=runs/E029-ce; until grep -q "E029 score ok\|CHAIN STOP" runs/E029-chain.log; do sleep 60; done; grep -q "E029 score ok" runs/E029-chain.log || { echo "CHAIN STOP: E029 not ok"; exit 1; }
else BM=runs/E027-ce-large/model; BCE=runs/E027-ce; fi
[ -f $BM/train_pairs.parquet ] || cp runs/E027-ce-large/model/train_pairs.parquet $BM/
echo "E030 base $BASE ($BM), start $(date +%H:%M)"
$PY -m monitor.launch --run E030-selftrain --quiet -- $PY -m src.er_selftrain train \
    --probs runs/E015/sub_E023b_st/test_probs_stage2.parquet --country France US India --base $BM \
    --out runs/E030-ce-st/model --n-pos 150000 --n-neg 200000 --batch 32 --lr 1e-5
ok E030-selftrain || { echo "CHAIN STOP: E030 self-train failed"; exit 1; }
echo "E030 train ok $(date +%H:%M)"
$PY -m monitor.launch --run E030-score-fr --quiet -- $PY -m src.er_selftrain score --model runs/E030-ce-st/model \
    --country France --only-scored --base-ce $BCE --out runs/E030-ce-fr --batch 256
ok E030-score-fr || { echo "CHAIN STOP: E030 France scoring failed"; exit 1; }
$PY -m monitor.launch --run E030-score-usin --quiet -- $PY -m src.er_selftrain score --model runs/E030-ce-st/model \
    --country US India --uncertain 0.02 0.98 --only-scored --base-ce runs/E030-ce-fr --out runs/E030-ce --batch 256
ok E030-score-usin || { echo "CHAIN STOP: E030 US/India scoring failed"; exit 1; }
echo "E030 scoring ok $(date +%H:%M)"
until [ -f runs/E027-stage2/exit.json ] && { [ -f runs/E029-final/exit.json ] || grep -q "CHAIN STOP" runs/E029-final-chain.log; } && [ "$(stage2_running)" = 0 ]; do sleep 60; done
mkdir -p runs/frames/E030_final && cp runs/frames/E027/train_frame.parquet runs/frames/E030_final/
$PY -c "import json; k=json.load(open('runs/frames/E027/frames.json')); k['ce_dir']=['runs/E015-ce','runs/E020-ce','runs/E022-llmce-full','runs/E030-ce']; json.dump(k, open('runs/frames/E030_final/frames.json','w'))"
echo "E030-final stage 2 start $(date +%H:%M)"
$PY -m monitor.launch --run E030-final --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E030-ce --fit-folds 0 --tag E030_final \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E030_final --out submissions/sub_E030_final
ok E030-final || { echo "CHAIN STOP: E030-final stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E030_final && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E030_final/
for v in sub_E030_final sub_E030_final_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E030_final/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
  cmp -s submissions/sub_E023b_full/candidate_pairs.tsv submissions/$v/candidate_pairs.tsv && echo "$v candidates identical to sub-12" || echo "WARNING: $v candidates differ from sub-12"
done
$PY scripts/cand_recall.py --tag E030_final --sub submissions/sub_E030_final_unseen
echo "E030-final DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/sub_E030_final/stage2.json')); p=json.load(open('runs/E015/sub_E030_final/predict.json')); print('dev', r['dev_f05'], r['dev_by_country'], '| mean matches', round(p['mean_matches'],3))")"
