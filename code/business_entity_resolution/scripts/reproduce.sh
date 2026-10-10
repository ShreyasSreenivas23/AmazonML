#!/usr/bin/env bash
# =====================================================================================================================
# End-to-end reproduction of the final submission (team SHSHSHSH, Amazon ML Challenge 2026, business entity resolution)
#
#   bash scripts/reproduce.sh                      # from the repo root (Linux / WSL2 / Git Bash on Windows)
#
# Produces output/matching_results.tsv and output/candidate_pairs.tsv (plus output_unseen/, see step 14), then runs
# both validators. Every step reads the outputs of the steps before it; nothing is downloaded after step 0.
#
# Reference machine: 1x RTX 5080 16 GB (bf16), 64 GB RAM, 32 CPU threads, ~200 GB free disk. Times below are wall
# clock on that machine. Total ~24 h of wall clock, ~20 GPU-hours (the CPU-only steps are short).
#
# Knobs (environment variables):
#   PY=python                  interpreter of the venv built from requirements.txt
#   SKIP_DOWNLOAD=1            skip step 0 when the three base models are already in the Hugging Face cache
#   E008_FAITHFUL=1            1 = train cross-encoder A on the E007 stage-1 chunks exactly as in the competition
#                              (+~2.5 h); 0 = train it on the E015 chunks instead (same folds 1-2, faster, not
#                              bit-identical to what we ran)
#   LLM_BAND=core              core = reranker feature on the E016-CE band (0.1, 0.9) only (runs/E022-llmce);
#                              full = also the wider band (0.02, 0.1] U [0.9, 0.98) (runs/E022-llmce-full, +~5 h GPU)
#                              (default full = the final submission)
#   FINAL_VARIANT=unseen       base = global decision threshold for every country; unseen = the stricter threshold
#                              (0.85) for S1 countries absent from train, written by the SAME stage-2 run
#                              (default unseen = the final submission)
#   FINAL_STACK=e034           e023b_st | e029 | e030 | e034 | e037: CE stack of the final stage 2 (steps 12c-12j)
#
# Most steps are resumable (embeddings, model training checkpoints, scoring shards, stage-1 chunks): after a crash,
# re-running this script skips finished work where the step supports it.
# =====================================================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
E008_FAITHFUL="${E008_FAITHFUL:-1}"
LLM_BAND="${LLM_BAND:-full}"
FINAL_VARIANT="${FINAL_VARIANT:-unseen}"
FINAL_STACK="${FINAL_STACK:-e034}"
DATA="${DATA_DIR:-data}"
export PYTHONUNBUFFERED=1

step() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

[ -f "$DATA/dataset/train/train_source1.tsv" ] && [ -f "$DATA/dataset/test/test_source1.tsv" ] || {
  echo "Put the organisers' files at $DATA/dataset/{train,test}/*.tsv first (see README.md)."; exit 1; }

# ---------------------------------------------------------------------------------------------------------------------
# 0. Base models into the local Hugging Face cache (the ONLY network access; ~11 GB). Afterwards the pipeline runs
#    offline. Revisions are the snapshots we used; a mismatch is reported (the models are then still usable).
#    in: -   out: ~/.cache/huggingface/hub   time: ~10 min (bandwidth-bound)
# ---------------------------------------------------------------------------------------------------------------------
if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  step "0. download base models"
  "$PY" - <<'EOF'
from pathlib import Path
from huggingface_hub import snapshot_download
PINNED = {  # repo -> snapshot (commit) used for the competition run
    "intfloat/multilingual-e5-small": "614241f622f53c4eeff9890bdc4f31cfecc418b3",   # MIT, 118M
    "intfloat/multilingual-e5-base": "d128750597153bb5987e10b1c3493a34e5a4502a",    # MIT, 278M
    "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",   # MIT, 560M
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",                    # Apache-2.0, 4.0B
}
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "sentencepiece*"]
for repo, rev in PINNED.items():
    path = snapshot_download(repo, revision="main", allow_patterns=ALLOW)
    got = Path(path).name
    print(f"{repo}: {got}" + ("" if got == rev else f"  WARNING: competition run used {rev}"))
EOF
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # from here on: no network access at all

# ---------------------------------------------------------------------------------------------------------------------
# 1. Exact TSV load (tab, no quoting, all strings) -> parquet cache; row counts verified.
#    in: data/dataset/*   out: data/cache/{train,test}_s{1,2,3}.parquet + ground-truth pairs   time: ~1 min (CPU)
# ---------------------------------------------------------------------------------------------------------------------
step "1. data cache"
"$PY" -m src.er_data

# ---------------------------------------------------------------------------------------------------------------------
# 2. Normalisation v1 (anyascii transliteration, lower-case, punctuation/`null` cleanup, legal-suffix core name,
#    numbers, postcode) and v2 (street types expanded, dotted legal forms collapsed; small hand-written dictionaries).
#    in: data/cache/*.parquet   out: data/cache/*_norm.parquet, *_norm2.parquet   time: ~3 min (30 processes)
# ---------------------------------------------------------------------------------------------------------------------
step "2. normalise (v1 + v2)"
"$PY" -m src.er_normalize
"$PY" -m src.er_norm2

# ---------------------------------------------------------------------------------------------------------------------
# 3. Shared folds: 5 random folds over train S1 (seed 42) + a fixed 100k-S1 dev subset of fold 0.
#    in: train S1   out: data/cache/folds_s1_k5.parquet   time: <1 min
# ---------------------------------------------------------------------------------------------------------------------
step "3. folds"
"$PY" scripts/make_folds.py

# ---------------------------------------------------------------------------------------------------------------------
# 4. Off-the-shelf multilingual-e5-small embeddings, 3 views (name / address / name + address), all sources, both splits.
#    in: data/cache   out: data/cache/emb/<split>_s<k>_<view>_small.npy (float16, L2-normalised)   time: ~75 min GPU
# ---------------------------------------------------------------------------------------------------------------------
step "4. e5-small embeddings (3 views)"
for split in train test; do
  for view in name addr both; do
    "$PY" -m src.er_embed --split "$split" --sources 1 2 3 --view "$view" --model small
  done
done

# ---------------------------------------------------------------------------------------------------------------------
# 5. E014: fine-tune e5-small as a bi-encoder (in-batch InfoNCE, same-country batches) on 1.5M labelled train pairs of
#    folds 1-4 (fold 0 unseen), then embed every record with it (4th retrieval view "both_ft").
#    out: runs/E014-bienc/model, data/cache/emb/<split>_s<k>_both_ft.npy   time: train ~18 min + embed ~10 min GPU
# ---------------------------------------------------------------------------------------------------------------------
step "5. E014 bi-encoder + both_ft embeddings"
"$PY" -m src.er_biencoder train --out runs/E014-bienc/model --n 1500000
for split in train test; do
  "$PY" -m src.er_embed --split "$split" --sources 1 2 3 --view both --model runs/E014-bienc/model --tag ft
done

# ---------------------------------------------------------------------------------------------------------------------
# 6. Training chunks for cross-encoder A (E008). In the competition E008 was trained on the stage-1 chunks of E007
#    (3 views, stage 1 fit on folds 1-4). E008_FAITHFUL=1 rebuilds them; 0 reuses the E015 chunks of step 7.
#    out: runs/E007-fullpass/{train,test}_chunks   time: ~2.5 h (GPU retrieval + CPU LightGBM)
# ---------------------------------------------------------------------------------------------------------------------
if [ "$E008_FAITHFUL" = 1 ]; then
  step "6. E007 stage 1 (training chunks for cross-encoder A)"
  "$PY" -m src.er_fullpass stage1 --exp E007-fullpass --views name addr both --train-s1 200000 --prefilter 15
  E008_CHUNKS=runs/E007-fullpass/train_chunks
else
  E008_CHUNKS=runs/E015/train_chunks
fi

# ---------------------------------------------------------------------------------------------------------------------
# 7. E015 stage 1 = blocking + first filter. Exact GPU top-10 cosine search per view (name, addr, both, both_ft) and
#    per source (S2, S3) within the same country string (~62 candidates/S1) -> 32 pair features -> LightGBM fit on
#    200k S1 of fold 0 minus dev (4 internal OOF groups) -> every train and test S1 scored, top-15 kept per S1.
#    out: runs/E015/{train,test}_chunks/*.parquet (s1_id, cand_id, prob, y, features)   time: ~3 h
# ---------------------------------------------------------------------------------------------------------------------
step "7. E015 stage 1"
"$PY" -m src.er_fullpass stage1 --exp E015 --views name addr both both_ft --fit-pool fold0 --train-s1 200000

# ---------------------------------------------------------------------------------------------------------------------
# 8. Cross-encoder A (E008): multilingual-e5-small sequence-pair classifier, 2M pairs (40% positive, hard negatives by
#    stage-1 prob) from S1 folds 1-2 only; then scores every E015 pair of fold 0 (stage-2 fit pool + dev) and test.
#    out: runs/E008-ce/model, runs/E015-ce/{train_ce,test_ce}   time: train ~25 min, scoring ~1.4 h GPU
# ---------------------------------------------------------------------------------------------------------------------
step "8. cross-encoder A (E008)"
"$PY" -m src.er_crossenc train --chunks "$E008_CHUNKS" --out runs/E008-ce/model --n 2000000 --exclude-folds 0 3 4
"$PY" -m src.er_crossenc score --model runs/E008-ce/model --chunks runs/E015/train_chunks --split train \
    --out runs/E015-ce/train_ce --only-folds 0
"$PY" -m src.er_crossenc score --model runs/E008-ce/model --chunks runs/E015/test_chunks --split test \
    --out runs/E015-ce/test_ce

# ---------------------------------------------------------------------------------------------------------------------
# 9. Cross-encoder B (E016): multilingual-e5-base, 3M pairs from E015 chunks of folds 1-4 (fold 0 excluded), batch 64,
#    lr 2e-5, 1 epoch; scores fold 0 + test. Its score also defines the reranker band (step 10) and the final
#    candidate filter (step 13).
#    out: runs/E016-ce-base/model, runs/E016-ce/{train_ce,test_ce}   time: train ~2.4 h, scoring ~1.2 h GPU
# ---------------------------------------------------------------------------------------------------------------------
step "9. cross-encoder B (E016)"
"$PY" -m src.er_crossenc train --chunks runs/E015/train_chunks --out runs/E016-ce-base/model --n 3000000 \
    --exclude-folds 0 --model-name intfloat/multilingual-e5-base --batch 64 --lr 2e-5 --ckpt-every 2000
"$PY" -m src.er_crossenc score --model runs/E016-ce-base/model --chunks runs/E015/train_chunks --split train \
    --out runs/E016-ce/train_ce --only-folds 0
"$PY" -m src.er_crossenc score --model runs/E016-ce-base/model --chunks runs/E015/test_chunks --split test \
    --out runs/E016-ce/test_ce

# ---------------------------------------------------------------------------------------------------------------------
# 10. E021 reranker prompts: S1 record, candidate record and up to 3 competing S1s that also retrieved the candidate
#     (full population of the same split). Training prompts from folds 1-4 (111,920 rows); scoring prompts for every
#     fold-0 and test pair whose E016 score lies in (0.02, 0.98).
#     out: runs/E021-llm/data/{train,score_train,score_test}.parquet   time: ~6 min CPU
# ---------------------------------------------------------------------------------------------------------------------
step "10. E021 reranker prompts"
"$PY" -m src.er_llmrank build --out runs/E021-llm/data

# ---------------------------------------------------------------------------------------------------------------------
# 11. Subsets actually used (built ad hoc during the competition; this snippet reproduces them exactly: checked
#     against our copies on 26 Sep, same rows in the same order):
#     data10   = 40,000 training prompts (seed-0 sample) + scoring prompts with E016 score in (0.1, 0.9)
#                (154,855 fold-0 + 726,778 test pairs)
#     data_ext = the rest of the (0.02, 0.98) band, scoring prompts only (308,502 fold-0 + 1,039,401 test pairs)
#     time: ~2 min CPU
# ---------------------------------------------------------------------------------------------------------------------
step "11. reranker data subsets (data10, data_ext)"
"$PY" - <<'EOF'
import json
from pathlib import Path
import pandas as pd

src, core, ext = Path("runs/E021-llm/data"), Path("runs/E021-llm/data10"), Path("runs/E021-llm/data_ext")
core.mkdir(parents=True, exist_ok=True)
ext.mkdir(parents=True, exist_ok=True)
pd.read_parquet(src / "train.parquet").sample(40_000, random_state=0).reset_index(drop=True) \
    .to_parquet(core / "train.parquet", index=False)
for split in ("train", "test"):
    sc = pd.read_parquet(src / f"score_{split}.parquet")
    ce = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "ce_score"])
                    for f in sorted(Path(f"runs/E016-ce/{split}_ce").glob("*.parquet"))], ignore_index=True)
    m = sc.merge(ce, on=["s1_id", "cand_id"], how="left")          # left merge keeps the prompt order
    assert m.ce_score.notna().all(), f"E016 score missing for some {split} prompts"
    band = ((m.ce_score > 0.1) & (m.ce_score < 0.9)).to_numpy()
    m.loc[band, ["s1_id", "cand_id", "prompt"]].reset_index(drop=True).to_parquet(core / f"score_{split}.parquet", index=False)
    m.loc[~band, ["s1_id", "cand_id", "prompt"]].reset_index(drop=True).to_parquet(ext / f"score_{split}.parquet", index=False)
    print(f"{split}: core band {band.sum():,}  wider band {(~band).sum():,}")
(core / "build.json").write_text(json.dumps({"from": str(src), "train_rows": 40_000, "band": [0.1, 0.9],
                                             "band_ce": "runs/E016-ce"}), encoding="utf-8")
EOF

# ---------------------------------------------------------------------------------------------------------------------
# 12. E022: Qwen3-4B + LoRA (r 16, all attention and MLP projections) trained as a Yes/No classifier on the 40k
#     prompts (batch 16, 1 epoch), then P(Yes) for the core band of fold 0 and test (batch 32: batch 64 spills GPU
#     memory on the longest prompts), mapped to a stage-2 feature (NaN outside the band).
#     out: runs/E022-llm/lora, runs/E022-llm/llm_{train,test}, runs/E022-llmce   time: train ~1.1 h, scoring ~3.3 h GPU
# ---------------------------------------------------------------------------------------------------------------------
step "12. E022 Qwen3-4B LoRA reranker (core band)"
"$PY" -m src.er_llmrank train --data runs/E021-llm/data10 --out runs/E022-llm/lora --batch 16 --ckpt-every 500
for split in train test; do
  "$PY" -m src.er_llmrank score --data runs/E021-llm/data10 --lora runs/E022-llm/lora --split "$split" \
      --out runs/E022-llm --batch 32
done
"$PY" -m src.er_llmrank to-ce --llm runs/E022-llm --out runs/E022-llmce
LLM_CE=runs/E022-llmce

# 12b. (LLM_BAND=full) the same LoRA on the wider band; core + wider scores merged into one feature directory.
#      out: runs/E022x-llm, runs/E022-llm-full, runs/E022-llmce-full   time: ~5 h GPU (estimate)
if [ "$LLM_BAND" = full ]; then
  step "12b. E022x reranker on the wider band"
  for split in train test; do
    "$PY" -m src.er_llmrank score --data runs/E021-llm/data_ext --lora runs/E022-llm/lora --split "$split" \
        --out runs/E022x-llm --batch 32
  done
  for split in train test; do
    mkdir -p runs/E022-llm-full/llm_$split
    for f in runs/E022-llm/llm_$split/*.parquet; do cp "$f" runs/E022-llm-full/llm_$split/core_$(basename "$f"); done
    for f in runs/E022x-llm/llm_$split/*.parquet; do cp "$f" runs/E022-llm-full/llm_$split/ext_$(basename "$f"); done
  done
  "$PY" -m src.er_llmrank to-ce --llm runs/E022-llm-full --out runs/E022-llmce-full
  LLM_CE=runs/E022-llmce-full
fi

# ---------------------------------------------------------------------------------------------------------------------
# 12c-12h. Final-day stack (27 Sep): self-training on test inputs (pseudo-labels from our own stage 2; no labels, no
#     external data), the e5-large cross-encoder, its francized continuation and its self-trained version.
#     FINAL_STACK=e023b_st | e029 | e030 selects the CE directories of the final stage 2 (step 13).
#     All learned models train on folds 1-4 + test pseudo-labels only; fold 0 stays clean for stage 2 and dev.
# ---------------------------------------------------------------------------------------------------------------------
PS1_PROBS=runs/E015/sub_E016/test_probs_stage2.parquet
step "12c. pseudo-label source 1: E016 stage 2 (test probabilities for E018/E020)"
"$PY" -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft --ce-dir runs/E015-ce runs/E016-ce \
    --fit-folds 0 --tag E016 --out runs/E016-sub
mkdir -p runs/E015/sub_E016 && cp runs/E015/test_probs_stage2.parquet "$PS1_PROBS"
step "12d. self-training: E018 (France, all rows) + E020 (US/India, CE-uncertain rows) -> runs/E020-ce"
"$PY" -m src.er_selftrain train --probs "$PS1_PROBS" --country France --base runs/E016-ce-base/model --out runs/E018-ce-fr/model
"$PY" -m src.er_selftrain score --model runs/E018-ce-fr/model --country France --base-ce runs/E016-ce --out runs/E018-ce
"$PY" -m src.er_selftrain train --probs "$PS1_PROBS" --country US India --base runs/E016-ce-base/model --out runs/E020-ce-usin/model
"$PY" -m src.er_selftrain score --model runs/E020-ce-usin/model --country US India --uncertain 0.02 0.98 \
    --base-ce runs/E018-ce --out runs/E020-ce
KEEP="--keep-prob 0.2 --keep-ce-dir runs/E016-ce --keep-ce 0.01"
step "12e. E027 multilingual-e5-large cross-encoder (folds 1-4), scored on the final candidate rows only"
"$PY" -m src.er_crossenc train --chunks runs/E015/train_chunks --out runs/E027-ce-large/model --n 1500000 \
    --exclude-folds 0 --model-name intfloat/multilingual-e5-large --batch 32 --lr 1.5e-5 --ckpt-every 2000
"$PY" -m src.er_crossenc score --model runs/E027-ce-large/model --chunks runs/E015/train_chunks --split train \
    --out runs/E027-ce/train_ce --only-folds 0 $KEEP --batch 256
"$PY" -m src.er_crossenc score --model runs/E027-ce-large/model --chunks runs/E015/test_chunks --split test \
    --out runs/E027-ce/test_ce $KEEP --batch 256
if [ "$FINAL_STACK" != e023b_st ]; then
  step "12f. E029: E027 continued on francized train pairs; unseen-country test rows re-scored -> runs/E029-ce"
  "$PY" scripts/francize_ce.py train
  "$PY" scripts/francize_ce.py score
fi
if [ "$FINAL_STACK" = e030 ] || [ "$FINAL_STACK" = e034 ] || [ "$FINAL_STACK" = e037 ]; then
  step "12g. pseudo-label source 2: E023b-ST stage 2 (test probabilities for E030)"
  "$PY" -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
      --ce-dir runs/E015-ce runs/E020-ce "$LLM_CE" --fit-folds 0 --tag E023b_st \
      --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
      --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --out runs/E023b-st-sub
  mkdir -p runs/E015/sub_E023b_st && cp runs/E015/test_probs_stage2.parquet runs/E015/sub_E023b_st/
  step "12h. E030: self-training of E029 (France + US/India pseudo-labels) -> runs/E030-ce"
  cp runs/E027-ce-large/model/train_pairs.parquet runs/E029-ce-fr/model/
  "$PY" -m src.er_selftrain train --probs runs/E015/sub_E023b_st/test_probs_stage2.parquet --country France US India \
      --base runs/E029-ce-fr/model --out runs/E030-ce-st/model --n-pos 150000 --n-neg 200000 --batch 32 --lr 1e-5
  "$PY" -m src.er_selftrain score --model runs/E030-ce-st/model --country France --only-scored --base-ce runs/E029-ce \
      --out runs/E030-ce-fr --batch 256
  "$PY" -m src.er_selftrain score --model runs/E030-ce-st/model --country US India --uncertain 0.02 0.98 --only-scored \
      --base-ce runs/E030-ce-fr --out runs/E030-ce --batch 256
fi
if [ "$FINAL_STACK" = e034 ] || [ "$FINAL_STACK" = e037 ]; then
  # 12i. OW04: listwise owner model (multilingual-e5-large, MIT). Reads a contested pool record with up to 6 competing
  #      S1s and predicts which one owns it. Trained on train groups with no fold-0 S1 involved (fold 0 stays clean).
  #      out: runs/OW04p (owner probability), runs/OW04m (owner margin)   time: prep ~10 min, train ~3 h, score ~1 h GPU
  step "12i. OW04 owner model -> runs/OW04p, runs/OW04m"
  "$PY" scripts/owner_pairs.py --out runs/OW04
  "$PY" -m src.er_owner prep --pairs runs/OW04/pairs.parquet --frame runs/OW04/fold0.parquet       --test-frame runs/OW04/test.parquet --out runs/OW04
  "$PY" -m src.er_owner train --dir runs/OW04 --model intfloat/multilingual-e5-large --epochs 2 --batch 32 --lr 2e-5       --n 200000 --bf16
  for split in train test; do "$PY" -m src.er_owner score --dir runs/OW04 --split $split --batch 256 --bf16; done
  "$PY" scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_p --out runs/OW04p
  "$PY" scripts/owner_to_ce.py --owner-dir runs/OW04 --col own_margin --out runs/OW04m
fi
RESCUE=""
if [ "$FINAL_STACK" = e037 ]; then
  # 12j. E034 stage 2 (the teacher), then E036: France-only self-training round 2 of the E030 CE on 450k confident French
  #      pseudo-labels from E034; France candidate rows re-scored. E037 = E034 inputs with runs/E030-ce -> runs/E036-ce plus
  #      the unseen-country candidate rescue (keep French pairs the France-adapted CE scores >= 0.5).
  step "12j. E034 teacher run, E036 France self-training round 2 -> runs/E036-ce"
  "$PY" -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
      --ce-dir runs/E015-ce runs/E020-ce "$LLM_CE" runs/E030-ce runs/OW04p runs/OW04m --fit-folds 0 --tag E034_ow04 \
      --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
      --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 --out runs/E034-sub
  mkdir -p runs/E015/sub_E034_ow04 && cp runs/E015/test_probs_stage2.parquet runs/E015/sub_E034_ow04/
  cp runs/E027-ce-large/model/train_pairs.parquet runs/E030-ce-st/model/
  "$PY" -m src.er_selftrain train --probs runs/E015/sub_E034_ow04/test_probs_stage2.parquet --country France \
      --base runs/E030-ce-st/model --out runs/E036f-ce-st/model --n-pos 200000 --n-neg 250000 --batch 32 --lr 1e-5
  "$PY" -m src.er_selftrain score --model runs/E036f-ce-st/model --country France --only-scored --base-ce runs/E030-ce \
      --out runs/E036-ce --batch 256
  RESCUE="--prune-unseen-ce-dir runs/E036-ce --prune-unseen-ce 0.5"
fi
case "$FINAL_STACK" in
  e037) FINAL_CE="runs/E015-ce runs/E020-ce $LLM_CE runs/E036-ce runs/OW04p runs/OW04m" ;;
  e034) FINAL_CE="runs/E015-ce runs/E020-ce $LLM_CE runs/E030-ce runs/OW04p runs/OW04m" ;;
  e030) FINAL_CE="runs/E015-ce runs/E020-ce $LLM_CE runs/E030-ce" ;;
  e029) FINAL_CE="runs/E015-ce runs/E020-ce $LLM_CE runs/E029-ce" ;;
  *)    FINAL_CE="runs/E015-ce runs/E020-ce $LLM_CE" ;;
esac

# ---------------------------------------------------------------------------------------------------------------------
# 13. Final stage 2 + decisions + both TSVs, in ONE run.
#     - features: stage-1 features + both cross-encoders + reranker + cluster (sibling) features + full-population
#       competition features (margin over the candidate's best other S1) on 5 name similarities + v2-normalised
#       similarities + label-free name rarity; competitor density on train thinned to test-like (--comp-keep 0.78)
#     - LightGBM fit on fold-0 S1s outside dev (4 internal OOF groups); dev F0.5 printed; decision rule and threshold
#       chosen on OOF (global threshold vs expected-F0.5, each S2/S3 record assigned to at most one S1)
#     - final candidate set = stage-1 prob >= 0.5 OR E016 CE >= 0.05 (dev: 3.83 per S1, pair recall 0.991); the
#       model is fitted and applied on exactly these rows, so candidate_pairs.tsv is its inference set
#     - --unseen-threshold 0.85 also writes output_unseen/: threshold 0.85 for S1 countries not present in train
#       (derived from the data; nothing is hard-coded), everything else identical
#     out: output/{matching_results,candidate_pairs}.tsv, output_unseen/, runs/E015/{stage2,predict}.json
#     time: ~1.5 h CPU, peak ~50 GB RAM
# ---------------------------------------------------------------------------------------------------------------------
step "13. final stage 2 ($FINAL_STACK) -> output/"
"$PY" -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
    --ce-dir $FINAL_CE --fit-folds 0 --tag FINAL \
    --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
    --prune-eps 0.5 --prune-ce-dir runs/E016-ce --prune-ce 0.05 $RESCUE --unseen-threshold 0.85 \
    --out output
if [ "$FINAL_VARIANT" = unseen ]; then
  cp output_unseen/matching_results.tsv output_unseen/candidate_pairs.tsv output/
fi

# ---------------------------------------------------------------------------------------------------------------------
# 14. Validators: ours (format + matches subset of candidates) and the organisers' (stdlib; place it at
#     data/utils/validate_submission.py from student_resource/utils/). time: ~2 min
# ---------------------------------------------------------------------------------------------------------------------
step "14. validate output/"
"$PY" -m src.er_submission validate --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv \
    --test-dir "$DATA/dataset/test"
if [ -f "$DATA/utils/validate_submission.py" ]; then
  "$PY" "$DATA/utils/validate_submission.py" --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir "$DATA/dataset/test" --check-ids
else
  echo "organisers' validator not found at $DATA/utils/validate_submission.py (skipped)"
fi
step "done: output/matching_results.tsv, output/candidate_pairs.tsv"
