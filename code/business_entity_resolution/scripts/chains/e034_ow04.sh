#!/usr/bin/env bash
# E034: E033 + the listwise owner model OW04 (OW04p owner probability, OW04m owner margin; multilingual-e5-large,
# MIT, trained on folds 1-4 groups only, no self-training). Appended at the END of E033's --ce-dir so existing CE column
# names are unchanged. Same candidate filter (3.83/S1) -> candidate_pairs.tsv must be byte-identical to E033's.
# Gate (all must hold, else ship E033): dev >= 0.99119, OOF >= 0.99100, US >= 0.99026, India >= 0.99185, validators PASS.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
$PY scripts/check_ce_alignment.py runs/OW04p runs/OW04m || { echo "CHAIN STOP: OW04 alignment failed"; exit 1; }
until [ -f runs/E031-final/exit.json ] || grep -q "CHAIN STOP" runs/E031-chain.log; do sleep 60; done
until [ "$(stage2_running)" = 0 ]; do sleep 60; done
echo "E034 stage 2 start $(date +%H:%M)"
$PY -m monitor.launch --run E034-ow04 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E030-ce runs/OW04p runs/OW04m --fit-folds 0 --tag E034_ow04 \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 --unseen-threshold 0.85 \
    --frames runs/frames/E034_ow04 --out submissions/sub_E034_ow04
grep -q '"returncode": 0' runs/E034-ow04/exit.json || { echo "CHAIN STOP: E034 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E034_ow04 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E034_ow04/
for v in sub_E034_ow04 sub_E034_ow04_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E034_ow04/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
cmp -s submissions/sub_E033_e030_c383_unseen/candidate_pairs.tsv submissions/sub_E034_ow04_unseen/candidate_pairs.tsv && echo "candidates byte-identical to E033" || echo "WARNING: candidates differ from E033"
$PY scripts/cand_recall.py --tag E034_ow04 --sub submissions/sub_E034_ow04_unseen --frames runs/frames/E034_ow04 --prune-eps 0.5 --prune-ce 0.05
$PY - <<'PYEOF'
import json
a = json.load(open("runs/E015/sub_E033_e030_c383/stage2.json")); b = json.load(open("runs/E015/sub_E034_ow04/stage2.json"))
g = {"dev>=0.99119": b["dev_f05"] >= a["dev_f05"] + 0.0002, "OOF>=E033": b["stage2_oof"] >= a["stage2_oof"],
     "US>=E033-0.0001": b["dev_by_country"]["US"] >= a["dev_by_country"]["US"] - 0.0001,
     "India>=E033-0.0001": b["dev_by_country"]["India"] >= a["dev_by_country"]["India"] - 0.0001}
print("E033 dev", round(a["dev_f05"], 5), "OOF", round(a["stage2_oof"], 5), a["dev_by_country"])
print("E034 dev", round(b["dev_f05"], 5), "OOF", round(b["stage2_oof"], 5), b["dev_by_country"])
print("GATE", "PASS" if all(g.values()) else "FAIL", g)
PYEOF
echo "E034 DONE $(date +%H:%M)"
