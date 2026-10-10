#!/usr/bin/env bash
# E020 = E019 features + France (E018) and US/India (this) self-trained CE on test. GPU after E018; stage 2 after
# E019's dev part (reuses its cached train frame). The submission for 26 Sep evening.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
ok() { grep -q '"returncode": 0' "runs/$1/exit.json" 2>/dev/null; }
until [ -f runs/E018-score/exit.json ]; do sleep 30; done
ok E018-score || { echo "CHAIN STOP: E018 failed"; exit 1; }
$PY -m monitor.launch --run E020-selftrain --quiet -- $PY -m src.er_selftrain train \
    --probs runs/E015/sub_E016/test_probs_stage2.parquet --country US India --base runs/E016-ce-base/model \
    --out runs/E020-ce-usin/model
ok E020-selftrain || { echo "CHAIN STOP: E020 self-train failed"; exit 1; }
$PY -m monitor.launch --run E020-score --quiet -- $PY -m src.er_selftrain score --model runs/E020-ce-usin/model \
    --country US India --uncertain 0.02 0.98 --base-ce runs/E018-ce --out runs/E020-ce
ok E020-score || { echo "CHAIN STOP: E020 scoring failed"; exit 1; }
echo "E020 CE ok $(date +%H:%M)"
until [ -f runs/E019-dev/exit.json ]; do sleep 30; done
ok E019-dev || { echo "CHAIN STOP: E019 dev failed"; exit 1; }
# train frame is identical (train_ce is the same E016 copy in E018-ce and E020-ce): reuse E019's, retagged for E020
mkdir -p runs/frames/E020 && cp runs/frames/E019/train_frame.parquet runs/frames/E020/
$PY -c "import json; k=json.load(open('runs/frames/E019/frames.json')); k['ce_dir']=['runs/E015-ce','runs/E020-ce']; json.dump(k, open('runs/frames/E020/frames.json','w'))"
COMP=$($PY -c "import json; c=json.load(open('runs/frames/E020/frames.json'))['comp_cols']; print(('--comp-cols ' + ' '.join(c)) if c else '')")
$PY -m monitor.launch --run E020-test --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce --fit-folds 0 --tag E020 $COMP --frames runs/frames/E020 --out submissions/sub_E020
ok E020-test || { echo "CHAIN STOP: E020 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E020 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/sub_E020/
echo "E020 CHAIN DONE $(date +%H:%M)"
