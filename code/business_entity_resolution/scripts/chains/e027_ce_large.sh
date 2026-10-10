#!/usr/bin/env bash
# E027 (HANDOFF step 0): our own multilingual-e5-large cross-encoder (MIT, 560M), trained on folds 1-4 only (fold 0 unseen),
# scored ONLY on the final candidate rows (stage-1 >= 0.2 OR E016 CE >= 0.01) of fold 0 + test, then stage 2 = E023a +
# this CE as one more feature. Kept only if dev at test-like density improves; candidate_pairs.tsv must stay
# byte-identical to E023a-core (same filter). Deadline ~15:00, else skipped.
# GPU waits for E022x (previous chat); stage 2 waits until no other stage 2 is running (RAM 64 GB: one at a time).
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
FILTER="--keep-prob 0.2 --keep-ce-dir runs/E016-ce --keep-ce 0.01"

until grep -q "E022x CHAIN DONE\|CHAIN STOP" runs/E022x-chain.log 2>/dev/null; do sleep 60; done
grep -q "E022x CHAIN DONE" runs/E022x-chain.log || { echo "CHAIN STOP: E022x did not finish (GPU state unknown)"; exit 1; }
# (a) backbone (no-op if already cached): safetensors + tokenizer only
$PY -c "from huggingface_hub import snapshot_download as s; print(s('intfloat/multilingual-e5-large', allow_patterns=['config.json','model.safetensors','sentencepiece.bpe.model','special_tokens_map.json','tokenizer.json','tokenizer_config.json']))" \
    || { echo "CHAIN STOP: e5-large download failed"; exit 1; }
echo "E027 start train $(date +%H:%M)"
# (b) train (resumable: rerun continues from runs/E027-ce-large/model/ckpt/last.pt)
$PY -m monitor.launch --run E027-ce-large-train --quiet -- $PY -m src.er_crossenc train --chunks runs/E015/train_chunks \
    --out runs/E027-ce-large/model --n 1500000 --exclude-folds 0 --model-name intfloat/multilingual-e5-large \
    --batch 32 --lr 1.5e-5 --ckpt-every 2000
ok E027-ce-large-train || { echo "CHAIN STOP: E027 train failed"; exit 1; }
echo "E027 train ok $(date +%H:%M)"
# (c) score only the final candidate rows (others NaN, row-aligned with the stage-1 chunks)
$PY -m monitor.launch --run E027-score-train --quiet -- $PY -m src.er_crossenc score --model runs/E027-ce-large/model \
    --chunks runs/E015/train_chunks --split train --out runs/E027-ce/train_ce --only-folds 0 $FILTER --batch 256
ok E027-score-train || { echo "CHAIN STOP: E027 score train failed"; exit 1; }
$PY -m monitor.launch --run E027-score-test --quiet -- $PY -m src.er_crossenc score --model runs/E027-ce-large/model \
    --chunks runs/E015/test_chunks --split test --out runs/E027-ce/test_ce $FILTER --batch 256
ok E027-score-test || { echo "CHAIN STOP: E027 score test failed"; exit 1; }
echo "E027 scoring ok $(date +%H:%M)"
# (d) stage 2 = E023a + E027; reranker band = the better of core / full (G0), by dev at test-like density
until [ "$(stage2_running)" = 0 ]; do sleep 60; done
LLM=$($PY -c "import json,os; f='runs/E015/sub_E023b_full/stage2.json'; c=json.load(open('runs/E015/sub_E023a_core/stage2.json'))['dev_f05']; print('runs/E022-llmce-full' if os.path.exists(f) and json.load(open(f))['dev_f05'] >= c else 'runs/E022-llmce')")
echo "E027 stage 2 with $LLM $(date +%H:%M)"
$PY -m monitor.launch --run E027-stage2 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E016-ce $LLM runs/E027-ce --fit-folds 0 --tag E027 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
    --frames runs/frames/E027 --out submissions/sub_E027
ok E027-stage2 || { echo "CHAIN STOP: E027 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E027 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E027/
for v in sub_E027 sub_E027_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E027/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
cmp -s submissions/sub_E023a_core/candidate_pairs.tsv submissions/sub_E027/candidate_pairs.tsv && echo "candidate_pairs identical to E023a-core" || echo "WARNING: candidate_pairs differ from E023a-core"
$PY scripts/cand_recall.py --tag E027 --sub submissions/sub_E027 --frames runs/frames/E027
echo "E027 DONE $(date +%H:%M): $($PY -c "import json; r=json.load(open('runs/E015/sub_E027/stage2.json')); print(r['dev_f05'], r['dev_by_country'])")"
