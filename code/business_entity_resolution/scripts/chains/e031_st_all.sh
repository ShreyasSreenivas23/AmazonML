#!/usr/bin/env bash
# E031: full self-training reach for US/India, MEASURED on dev. The E030 self-trained large CE (test pseudo-labels for
# France/US/India + folds 1-4 train pairs; fold 0 never seen) scores ALL final candidate rows of fold 0 and test, and
# is added as one more stage-2 feature on top of E030-final's inputs. Dev (US/India labels) shows whether it helps.
# Same candidate filter -> candidate_pairs.tsv byte-identical to sub-12. Used in the final only if dev > E030-final.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
KEEP="--keep-prob 0.2 --keep-ce-dir runs/E016-ce --keep-ce 0.01"
until grep -q "E030 scoring ok\|CHAIN STOP" runs/E030-chain.log 2>/dev/null; do sleep 60; done
grep -q "E030 scoring ok" runs/E030-chain.log || { echo "CHAIN STOP: E030 not ok"; exit 1; }
echo "E031 scoring start $(date +%H:%M)"
$PY -m monitor.launch --run E031-score-train --quiet -- $PY -m src.er_crossenc score --model runs/E030-ce-st/model \
    --chunks runs/E015/train_chunks --split train --out runs/E031-ce/train_ce --only-folds 0 $KEEP --batch 256
ok E031-score-train || { echo "CHAIN STOP: E031 train scoring failed"; exit 1; }
$PY -m monitor.launch --run E031-score-test --quiet -- $PY -m src.er_crossenc score --model runs/E030-ce-st/model \
    --chunks runs/E015/test_chunks --split test --out runs/E031-ce/test_ce $KEEP --batch 256
ok E031-score-test || { echo "CHAIN STOP: E031 test scoring failed"; exit 1; }
echo "E031 scoring ok $(date +%H:%M)"
until [ -f runs/E030-final/exit.json ] || grep -q "CHAIN STOP" runs/E030-chain.log; do sleep 60; done
until [ "$(stage2_running)" = 0 ]; do sleep 60; done
echo "E031 stage 2 start $(date +%H:%M)"
$PY -m monitor.launch --run E031-final --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E030-ce runs/E031-ce --fit-folds 0 --tag E031_final \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E031_final --out submissions/sub_E031_final
ok E031-final || { echo "CHAIN STOP: E031-final stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E031_final && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E031_final/
for v in sub_E031_final sub_E031_final_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E031_final/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
  cmp -s submissions/sub_E023b_full/candidate_pairs.tsv submissions/$v/candidate_pairs.tsv && echo "$v candidates identical to sub-12" || echo "WARNING: $v candidates differ from sub-12"
done
$PY scripts/cand_recall.py --tag E031_final --sub submissions/sub_E031_final_unseen --frames runs/frames/E031_final
echo "E031-final DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/sub_E031_final/stage2.json')); print('dev', r['dev_f05'], r['dev_by_country'])") | E030-final $($PY -c "import json; r=json.load(open('runs/E015/sub_E030_final/stage2.json')); print(r['dev_f05'])" 2>/dev/null)"
