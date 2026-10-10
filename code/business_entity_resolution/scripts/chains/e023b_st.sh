#!/usr/bin/env bash
# E023b-ST: E023b-full with the self-trained CE scores on test (runs/E020-ce) instead of plain E016. Train side identical
# (E020-ce/train_ce is a copy of E016's), so E023b-full's cached train frame is reused. Candidate filter is the same
# (plain E016 >= 0.01 OR stage-1 >= 0.2) -> identical candidate_pairs.tsv. Ship only if organisers allow self-training.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
until [ -f runs/E023b-full/exit.json ] || [ -f runs/E015/sub_E023b_full/RECOVERED ]; do sleep 60; done
grep -q '"returncode": 0' runs/E023b-full/exit.json 2>/dev/null || [ -f runs/E015/sub_E023b_full/RECOVERED ]     || { echo "CHAIN STOP: E023b-full failed"; exit 1; }
mkdir -p runs/frames/E023b_st && cp runs/frames/E023b_full/train_frame.parquet runs/frames/E023b_st/
$PY -c "import json; k=json.load(open('runs/frames/E023b_full/frames.json')); k['ce_dir']=['runs/E015-ce','runs/E020-ce','runs/E022-llmce-full']; json.dump(k, open('runs/frames/E023b_st/frames.json','w'))"
$PY -m monitor.launch --run E023b-st --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full --fit-folds 0 --tag E023b_st \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E023b_st --out submissions/sub_E023b_st
grep -q '"returncode": 0' runs/E023b-st/exit.json || { echo "CHAIN STOP: E023b-st failed"; exit 1; }
mkdir -p runs/E015/sub_E023b_st && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E023b_st/
cmp -s submissions/sub_E023b_full/candidate_pairs.tsv submissions/sub_E023b_st/candidate_pairs.tsv && echo "candidate_pairs identical to E023b-full" || echo "WARNING: candidate_pairs differ from E023b-full"
echo "E023b-st DONE $(date +%H:%M)"
for v in sub_E023b_st sub_E023b_st_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv       --test-dir data/dataset/test --check-ids > runs/E015/sub_E023b_st/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
