#!/usr/bin/env bash
# E038: France self-training round 3. Continue the round-2 CE (E036) on 450k confident French pseudo-labels from E037;
# re-score French candidate rows; stage 2 = E037 with runs/E036-ce -> runs/E038-ce (also as the rescue CE). Train side
# unchanged (E038-ce/train_ce = E036-ce's copy) -> E036's train frame reused, test frame rebuilt. Ship only if French
# uncertainty drops further, the drift guard vs E037 passes, validators PASS and candidates stay ~3.98/S1.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
echo "E038 train start $(date +%H:%M)"
$PY -m monitor.launch --run E038-selftrain --quiet -- $PY -m src.er_selftrain train \
    --probs runs/E015/sub_E037_rescue/test_probs_stage2.parquet --country France --base runs/E036f-ce-st/model \
    --out runs/E038-ce-st/model --n-pos 200000 --n-neg 250000 --batch 32 --lr 1e-5 --seed 1
ok E038-selftrain || { echo "CHAIN STOP: E038 self-train failed"; exit 1; }
echo "E038 train ok $(date +%H:%M)"
$PY -m monitor.launch --run E038-score-fr --quiet -- $PY -m src.er_selftrain score --model runs/E038-ce-st/model \
    --country France --only-scored --base-ce runs/E036-ce --out runs/E038-ce --batch 256
ok E038-score-fr || { echo "CHAIN STOP: E038 France scoring failed"; exit 1; }
echo "E038 scoring ok $(date +%H:%M)"
mkdir -p runs/frames/E038_st3 && cp runs/frames/E036_st2/train_frame.parquet runs/frames/E038_st3/
$PY -c "import json; k=json.load(open('runs/frames/E036_st2/frames.json')); k['ce_dir']=[d.replace('runs/E036-ce','runs/E038-ce') for d in k['ce_dir']]; json.dump(k, open('runs/frames/E038_st3/frames.json','w'))"
$PY -m monitor.launch --run E038-final --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E038-ce runs/OW04p runs/OW04m --fit-folds 0 --tag E038_st3 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 --prune-eps 0.5 \
    --prune-ce-dir runs/E016-ce --prune-ce 0.05 --prune-unseen-ce-dir runs/E038-ce --prune-unseen-ce 0.5 --unseen-threshold 0.85 \
    --frames runs/frames/E038_st3 --out submissions/sub_E038_st3
ok E038-final || { echo "CHAIN STOP: E038 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E038_st3 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E038_st3/
for v in sub_E038_st3 sub_E038_st3_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E038_st3/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
$PY scripts/cand_recall.py --tag E038_st3 --sub submissions/sub_E038_st3_unseen --frames runs/frames/E038_st3 --prune-eps 0.5 --prune-ce 0.05
$PY scripts/france_drift.py --base runs/E015/sub_E037_rescue --new runs/E015/sub_E038_st3
echo "E038 DONE $(date +%H:%M): dev $($PY -c "import json; r=json.load(open('runs/E015/sub_E038_st3/stage2.json')); print(round(r['dev_f05'],5))")"
