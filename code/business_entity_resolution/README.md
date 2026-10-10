# Business Entity Resolution: how to reproduce

Team **SHSHSHSH**, Amazon ML Challenge 2026.
This file is `code/business_entity_resolution/README.md` in the submission zip. It regenerates
`output/matching_results.tsv` and `output/candidate_pairs.tsv` from the organisers' train/test files, using only
the code in this folder, the pinned packages in `requirements.txt` and four open base models (MIT / Apache-2.0).

## Overview

A retrieve, filter, score, decide pipeline (details in `Documentation_template.md`):

1. **Normalise** names and addresses (transliterate every script to ASCII, clean legal suffixes, numbers, postcodes).
2. **Block** within the same `country` string with exact GPU nearest-neighbour search over four embedding views
   (multilingual-e5-small on name, address and name + address, plus a bi-encoder fine-tuned on train matches):
   ~62 candidates per S1.
3. **Stage 1**: a LightGBM pair model on 32 similarity/context features keeps the top 15 per S1.
4. **Scorers**: two fine-tuned cross-encoders (e5-small, e5-base) score every kept pair; a Qwen3-4B LoRA reranker
   scores the uncertain pairs with the competing S1s shown in its prompt.
5. **Final candidate set**: stage-1 prob >= 0.2 OR cross-encoder B >= 0.01, about 4.7 candidates per S1. This is
   exactly the set the final model runs on and what `candidate_pairs.tsv` contains.
6. **Stage 2**: a LightGBM on 80 features, including full-population competition features (how the candidate record
   scores against the other S1s). Then an F0.5-tuned decision layer where each S2/S3 record goes to at most one S1.

## Hardware and runtime

Reference machine: 1x NVIDIA RTX 5080 16 GB (bf16), 64 GB RAM, 32 CPU threads, Windows 11 with Git Bash (WSL2/Linux
works the same). About **24 h of wall clock and ~20 GPU-hours** end to end. Peak RAM ~50 GB (stage 2). About 200 GB of
free disk (embedding matrices ~140 GB, stage-1 chunks and caches ~25 GB).

| # | Step | Module | Device | Time |
|---|---|---|---|---|
| 0 | Download 4 base models (~11 GB), then offline | `huggingface_hub` | net | ~10 min |
| 1 | TSV to parquet cache, row counts checked | `src.er_data` | CPU | ~1 min |
| 2 | Normalisation v1 + v2 | `src.er_normalize`, `src.er_norm2` | CPU | ~3 min |
| 3 | Folds (5 x S1, seed 42) + 100k dev S1 | `scripts/make_folds.py` | CPU | <1 min |
| 4 | e5-small embeddings, 3 views, train + test | `src.er_embed` | GPU | ~75 min |
| 5 | E014 bi-encoder train + `both_ft` embeddings | `src.er_biencoder`, `src.er_embed` | GPU | ~30 min |
| 6 | E007 stage 1 (training chunks for cross-encoder A; optional, see below) | `src.er_fullpass stage1` | GPU+CPU | ~2.5 h |
| 7 | E015 stage 1: 4-view retrieval, LightGBM, top-15 | `src.er_fullpass stage1` | GPU+CPU | ~3 h |
| 8 | Cross-encoder A (E008, e5-small): train + score fold 0 and test | `src.er_crossenc` | GPU | ~1.8 h |
| 9 | Cross-encoder B (E016, e5-base): train + score fold 0 and test | `src.er_crossenc` | GPU | ~3.6 h |
| 10 | Reranker prompts (E021) | `src.er_llmrank build` | CPU | ~6 min |
| 11 | Prompt subsets `data10` / `data_ext` | inline in `reproduce.sh` | CPU | ~2 min |
| 12 | Qwen3-4B LoRA (E022): train + score core band + map to a feature | `src.er_llmrank` | GPU | ~4.4 h |
| 12b | Reranker on the wider band (only if `LLM_BAND=full`) | `src.er_llmrank` | GPU | ~5 h (estimate) |
| 13 | Final stage 2 + decisions + both TSVs | `src.er_fullpass stage2` | CPU | ~1.5 h |
| 14 | Our validator + the organisers' validator | `src.er_submission`, `validate_submission.py` | CPU | ~2 min |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # Python 3.11; torch 2.8.0 + CUDA 12.8 wheels (needed for sm_120 GPUs)
cp .env.example .env                 # DATA_DIR defaults to ./data
```

Data: put the organisers' files at `data/dataset/train/*.tsv` and `data/dataset/test/*.tsv`. For the organisers'
validator in step 14, copy `student_resource/utils/validate_submission.py` to `data/utils/`.

A CUDA GPU is needed in practice: without one, the code falls back to CPU, but embedding ~24M texts and the
cross-encoder/LLM scoring become impractically slow. LightGBM, rapidfuzz and the normalisation use 32 CPU threads.

## One command

```bash
bash scripts/reproduce.sh
```

Options (environment variables, defaults first):
- `E008_FAITHFUL=1|0`: `1` rebuilds the E007 stage-1 chunks that cross-encoder A was trained on in the competition
  (+2.5 h). `0` trains it on the E015 chunks instead. That uses the same folds (1-2) and keeps fold 0 unseen, but the
  weights are not identical to ours.
- `LLM_BAND=core|full`: reranker feature on the E016 band (0.1, 0.9) only, or also on the wider band
  (0.02, 0.1] U [0.9, 0.98). The final submission uses `full`.
- `FINAL_VARIANT=base|unseen`: `unseen` copies `output_unseen/` (threshold 0.85 for S1 countries absent from train,
  from the same stage-2 run) into `output/`. The final submission uses `unseen`.
- `SKIP_DOWNLOAD=1`: the base models are already cached.
- `PY=...`: the interpreter to use.

Most steps resume after a crash (embeddings, model checkpoints, scoring shards, stage-1 chunks), so re-running the
script continues where it stopped.

## Numbered steps (what `scripts/reproduce.sh` runs)

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # after step 0 (model download)
python -m src.er_data                                                     # 1
python -m src.er_normalize && python -m src.er_norm2                      # 2
python scripts/make_folds.py                                              # 3
for s in train test; do for v in name addr both; do                       # 4
  python -m src.er_embed --split $s --sources 1 2 3 --view $v --model small; done; done
python -m src.er_biencoder train --out runs/E014-bienc/model --n 1500000  # 5
for s in train test; do python -m src.er_embed --split $s --sources 1 2 3 --view both \
  --model runs/E014-bienc/model --tag ft; done
python -m src.er_fullpass stage1 --exp E007-fullpass --views name addr both --train-s1 200000 --prefilter 15   # 6
python -m src.er_fullpass stage1 --exp E015 --views name addr both both_ft --fit-pool fold0 --train-s1 200000  # 7
python -m src.er_crossenc train --chunks runs/E007-fullpass/train_chunks --out runs/E008-ce/model \
  --n 2000000 --exclude-folds 0 3 4                                                                            # 8
python -m src.er_crossenc score --model runs/E008-ce/model --chunks runs/E015/train_chunks --split train \
  --out runs/E015-ce/train_ce --only-folds 0
python -m src.er_crossenc score --model runs/E008-ce/model --chunks runs/E015/test_chunks --split test \
  --out runs/E015-ce/test_ce
python -m src.er_crossenc train --chunks runs/E015/train_chunks --out runs/E016-ce-base/model --n 3000000 \
  --exclude-folds 0 --model-name intfloat/multilingual-e5-base --batch 64 --lr 2e-5 --ckpt-every 2000          # 9
python -m src.er_crossenc score --model runs/E016-ce-base/model --chunks runs/E015/train_chunks --split train \
  --out runs/E016-ce/train_ce --only-folds 0
python -m src.er_crossenc score --model runs/E016-ce-base/model --chunks runs/E015/test_chunks --split test \
  --out runs/E016-ce/test_ce
python -m src.er_llmrank build --out runs/E021-llm/data                                                        # 10
# 11: data10 = 40k seed-0 training prompts + scoring prompts with E016 score in (0.1, 0.9); data_ext = the rest
python -m src.er_llmrank train --data runs/E021-llm/data10 --out runs/E022-llm/lora --batch 16 --ckpt-every 500 # 12
python -m src.er_llmrank score --data runs/E021-llm/data10 --lora runs/E022-llm/lora --split train --out runs/E022-llm --batch 32
python -m src.er_llmrank score --data runs/E021-llm/data10 --lora runs/E022-llm/lora --split test  --out runs/E022-llm --batch 32
python -m src.er_llmrank to-ce --llm runs/E022-llm --out runs/E022-llmce
python -m src.er_fullpass stage2 --exp E015 --views name addr both both_ft \
  --ce-dir runs/E015-ce runs/E016-ce runs/E022-llmce --fit-folds 0 --tag E023 \
  --comp-cols cos_name name_ratio name_jw name_full_tset name_tsort --norm2 --comp-keep 0.78 \
  --prune-eps 0.2 --prune-ce-dir runs/E016-ce --prune-ce 0.01 --unseen-threshold 0.85 \
  --frames runs/frames/E023 --out output                                                                       # 13
python -m src.er_submission validate --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir data/dataset/test                                          # 14
python data/utils/validate_submission.py --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv --test-dir data/dataset/test --check-ids
```

Step 13 prints the held-out dev F0.5 (100k fold-0 S1s that no model was trained on), the out-of-fold F0.5, the chosen
decision rule, and the test candidate and match statistics. `--out output` also writes `output_unseen/`.

## Determinism

Seeds are fixed everywhere: folds 42, model training 0, LightGBM 42, density thinning 11, prompt subsets 0. The CPU
part is deterministic given its inputs: rebuilding the stage-2 design matrices from scratch (run REGCHECK-E020)
reproduced the E020 dev F0.5 (0.990972) to every printed digit. The GPU steps run in bf16, and some CUDA kernels are
not bit-deterministic, so embeddings and cross-encoder/LLM scores can differ in the last bits and a few borderline
decisions can flip. We expect a full re-run to match our numbers within about 1e-4 dev F0.5, with near-identical
candidate sets. Keep LightGBM at 32 threads (`num_threads` in `src/er_model.py`), because other thread counts change
floating-point summation order. The cross-encoder A training chunks come from E007 (`E008_FAITHFUL=1`, the default).

## Models and licences

| Model / library | Licence | Parameters | Role | Trained on |
|---|---|---|---|---|
| intfloat/multilingual-e5-small | MIT | 118M | retrieval embeddings (3 views); base of the bi-encoder (E014) and cross-encoder A (E008) | fine-tuned only on the provided train data |
| intfloat/multilingual-e5-base | MIT | 278M | cross-encoder B (E016) | provided train data only |
| intfloat/multilingual-e5-large | MIT | 560M | cross-encoder C (E027/E029/E030) and owner model OW04 | provided train data only (+ self-training on test inputs, allowed by the organisers) |
| Qwen/Qwen3-4B | Apache-2.0 | 4.0B | LoRA Yes/No reranker (E022) | provided train data only (LoRA adapters) |
| LightGBM | MIT | n/a (trees) | stage-1 and stage-2 pair classifiers | provided train data |
| rapidfuzz, scikit-learn, pandas, numpy, pyarrow | MIT / BSD | n/a | string similarity, data handling | n/a |
| anyascii | ISC | n/a | Unicode to ASCII transliteration (character tables only) | n/a |
| torch, transformers, peft | BSD-3 / Apache-2.0 | n/a | training and inference | n/a |

Every model is MIT or Apache-2.0 and at most 8B parameters (the largest is 4.0B). Pinned snapshots are in
`scripts/reproduce.sh`, step 0.

## Compliance

- **No external data, APIs, lookups or geocoding.** The only network access is the one-time model download in step 0.
  After it, `HF_HUB_OFFLINE=1` is set and `src/` makes no network calls. No hosted LLM is used.
- **Only the provided training data** trains or tunes anything. Test records are used only unlabelled, at inference
  (retrieval and label-free population statistics such as how many S1s retrieve a record). An earlier experiment that
  fine-tuned on test pseudo-labels (E018/E020, self-training) is **not** part of this pipeline.
- **Country is an open set.** Blocking compares the `country` string for equality. No country value is hard-coded,
  filtered or one-hot encoded. The optional stricter threshold applies to "S1 countries not seen in train", computed
  from the data.
- **Hand-written normalisation only:** small generic dictionaries (legal-form words, street-type abbreviations such as
  St/Street and R/Rue, `&` to "and", `null` cleanup) in `src/er_normalize.py` and `src/er_norm2.py`. No
  postal/gazetteer/business datasets and no libpostal.

## Weights

Trained weights are not shipped (~8 GB with resumable checkpoints; ~2 GB of final weights: bi-encoder and
cross-encoder A ~0.47 GB each, cross-encoder B ~1.1 GB, LoRA adapters 137 MB). They are available on request.
`scripts/reproduce.sh` retrains all of them.

## File map

| Path | Purpose |
|---|---|
| `scripts/reproduce.sh` | end-to-end run (this README, steps 0-14) |
| `src/er_data.py` | exact TSV loading, parquet cache, ground-truth pairs |
| `src/er_normalize.py`, `src/er_norm2.py` | normalisation v1 (transliteration, core name, numbers, postcode) and v2 (street types, dotted legal forms) |
| `scripts/make_folds.py`, `src/cv.py` | shared S1 folds and the dev subset |
| `src/er_embed.py` | multilingual-e5 embeddings per view (resumable) |
| `src/er_biencoder.py` | bi-encoder fine-tuning (E014) |
| `src/er_pipeline.py`, `src/er_blocking.py` | same-country exact top-k retrieval, featurisation, decision-rule selection |
| `src/er_features.py` | stage-1 pair features (embedding cosines, rapidfuzz name/address, numbers, postcode, rank context) |
| `src/er_fullpass.py` | stage 1 over all S1s, stage 2 (features, candidate filter, LightGBM, decisions, outputs) |
| `src/er_stage2.py` | cluster (sibling) and full-population competition features |
| `src/er_crossenc.py` | cross-encoder training and scoring |
| `src/er_llmrank.py` | reranker prompts, LoRA training, scoring, mapping to a stage-2 feature |
| `src/er_model.py` | LightGBM with grouped out-of-fold training |
| `src/er_decide.py` | threshold and expected-F0.5 decisions with one-S1-per-record assignment |
| `src/metrics.py` | the official macro F0.5 per S1 entity (`er_f05`), unit-tested |
| `src/er_submission.py` | writes and validates both TSVs |
| `monitor/heartbeat.py` | progress heartbeat imported by long jobs (local file only, no network) |
| `src/er_selftrain.py` | self-training experiment (E018/E020); **not used** by `reproduce.sh` |
| `scripts/chains/*.sh` | the exact job chains run during the competition (Windows paths); `reproduce.sh` is their clean equivalent |
| `tests/` | pytest on synthetic data (metric, validator, features, decisions) |
