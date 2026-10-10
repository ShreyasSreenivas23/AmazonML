#!/usr/bin/env bash
# E035: E034 + CE03 (an e5-large pair cross-encoder trained on folds 1-4 only; MIT), appended at the END of E034's
# --ce-dir so existing CE column names are unchanged. Same candidate filter -> candidate_pairs.tsv identical to E033/E034.
# Gate vs E034 (all must hold, else keep E034): dev >= E034 + 0.0001, OOF >= E034, US/India not worse than E034 by
# more than 0.0001, validators PASS, candidates identical, France drift guard PASS.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
stage2_running() { powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'er_fullpass stage2' }).Count" | tr -d '\r'; }
until [ -d runs/CE03/test_ce ] && [ "$(ls runs/CE03/test_ce | wc -l)" = 9 ]; do sleep 30; done
$PY scripts/check_ce_alignment.py runs/CE03 || { echo "CHAIN STOP: CE03 alignment failed"; exit 1; }
until [ -f runs/E034-ow04/exit.json ] && grep -q "E034 DONE\|CHAIN STOP" runs/E034-chain.log; do sleep 30; done
grep -q '"returncode": 0' runs/E034-ow04/exit.json || { echo "CHAIN STOP: E034 failed, nothing to extend"; exit 1; }
until [ "$(stage2_running)" = 0 ]; do sleep 30; done
echo "E035 stage 2 start $(date +%H:%M)"
$PY -m monitor.launch --run E035-ce03 --quiet -- $PY -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir runs/E015-ce runs/E020-ce runs/E022-llmce-full runs/E030-ce runs/OW04p runs/OW04m runs/CE03 --fit-folds 0 \
    --tag E035_ce03 --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 --unseen-threshold 0.85 \
    --frames runs/frames/E035_ce03 --out submissions/sub_E035_ce03
grep -q '"returncode": 0' runs/E035-ce03/exit.json || { echo "CHAIN STOP: E035 stage 2 failed"; exit 1; }
mkdir -p runs/E015/sub_E035_ce03 && cp runs/E015/stage2.json runs/E015/predict.json runs/E015/test_probs_stage2.parquet runs/E015/sub_E035_ce03/
for v in sub_E035_ce03 sub_E035_ce03_unseen; do
  $PY data/utils/validate_submission.py --matching submissions/$v/matching_results.tsv --candidate submissions/$v/candidate_pairs.tsv \
      --test-dir data/dataset/test --check-ids > runs/E015/sub_E035_ce03/validate_$v.txt 2>&1 && echo "$v: organiser validator PASS" || echo "$v: organiser validator FAIL"
done
cmp -s submissions/sub_E033_e030_c383_unseen/candidate_pairs.tsv submissions/sub_E035_ce03_unseen/candidate_pairs.tsv && echo "candidates byte-identical to E033" || echo "WARNING: candidates differ from E033"
$PY scripts/france_drift.py --base runs/E015/sub_E034_ow04 --new runs/E015/sub_E035_ce03 | grep -E "DRIFT_GUARD|same_candidate"
$PY - <<'PYEOF'
import json
a = json.load(open("runs/E015/sub_E034_ow04/stage2.json")); b = json.load(open("runs/E015/sub_E035_ce03/stage2.json"))
g = {"dev>=E034+0.0001": b["dev_f05"] >= a["dev_f05"] + 0.0001, "OOF>=E034": b["stage2_oof"] >= a["stage2_oof"],
     "US>=E034-0.0001": b["dev_by_country"]["US"] >= a["dev_by_country"]["US"] - 0.0001,
     "India>=E034-0.0001": b["dev_by_country"]["India"] >= a["dev_by_country"]["India"] - 0.0001}
print("E034 dev", round(a["dev_f05"], 5), "OOF", round(a["stage2_oof"], 5), a["dev_by_country"])
print("E035 dev", round(b["dev_f05"], 5), "OOF", round(b["stage2_oof"], 5), b["dev_by_country"])
print("GATE", "PASS" if all(g.values()) else "FAIL", g)
PYEOF
echo "E035 DONE $(date +%H:%M)"
