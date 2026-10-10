# Business Entity Resolution: Approach Summary (team SHSHSHSH)

**Result:** public LB **0.98793** (final model E034), dev macro F0.5 0.99154 on 100k held-out Source-1 entities.
**Candidate set:** **3.98 candidates per S1** on test (6.9M pairs, reduction ratio 99.99996%), dev pair recall 0.991.
Full write-up: `Documentation_template.md`; code and one-command reproduction: `code/business_entity_resolution/`.

## 1. Approach
A cascade that goes from cheap, high-recall steps to expensive, precise ones. Only provided data is used; every
model is MIT/Apache and at most 4B parameters.

1. **Normalisation.** Unicode clean-up, anyascii transliteration (Indic scripts), a small hand-written list of legal
   suffixes and street types.
2. **Blocking (dense retrieval).** Same-country k-nearest-neighbour search over four multilingual-e5 embedding views
   (name, address, name+address, and a bi-encoder we fine-tuned on train matches): top-10 per view and source,
   ~62 pairs per S1, recall 0.9976. Exact GPU search here; an ANN index (FAISS IVF/HNSW) at billion scale.
3. **Stage-1 LightGBM** on 32 cheap features (embedding cosines, rapidfuzz name/address similarities, number
   agreement, rank context) keeps the top 15 pairs per S1.
4. **Candidate filter = `candidate_pairs.tsv`.** Keep a pair if stage-1 probability >= 0.5 or an e5-base cross-encoder
   scores >= 0.05: **3.98 pairs per S1**. Tightening from 4.71 cost only 0.00005 dev F0.5.
5. **Pair scorers.** Fine-tuned cross-encoders (multilingual-e5 small, base and large); a Qwen3-4B LoRA reranker whose
   prompt shows the competing S1s; and a **listwise owner model** that reads a contested record with up to 6
   competing S1s and predicts which one owns it.
6. **Stage-2 LightGBM matcher** fuses all scores with cluster (sibling) features and full-population competition
   features (our margin over the record's best other S1), computed at test-like competitor density.
7. **Decisions.** Threshold 0.75 (0.85 for the country absent from train), each S2/S3 record to at most one S1, and
   an empty list when nothing passes (singleton credit under F0.5).

**France (not in train).** Country is treated as an open set. The e5-large cross-encoder is adapted with francized
synthetic training pairs and self-training on confident test predictions (both built from provided data only).

**Validation.** Grouped folds by S1. Every learned upstream model trains on folds 1-4; the stage-2 matcher is fitted
on fold 0 (unseen by all of them), and dev is 100k fold-0 entities scored at test-like density. Both validators run on
every file, and a re-run from code reproduces the TSVs byte for byte.

## 2. Key experiments (dev F0.5 at test-like density / public LB)
| Step | Dev | Public LB |
|---|---:|---:|
| Retrieval + LightGBM + threshold (baseline) | 0.9476 | - |
| Full-population stage 2 with competition features | 0.9643 | 0.947598 |
| + cross-encoders, fine-tuned bi-encoder view, name competition features | 0.9910 | 0.985645 |
| + LLM reranker, test-like density, v2 normalisation (E023b) | 0.99100 | 0.98631 |
| + France-adapted e5-large CE, 3.98/S1 filter (E033) | 0.99099 | 0.987692 |
| **+ listwise owner model (E034, final)** | **0.99154** | **0.98793** |

What did not help: LightGBM tuning (saturated), importance weighting towards France's competition regime, extra
cross-encoders beyond e5-large, a second self-training round, and looser France thresholds (false merges).

## 3. Conclusion
Recall first, then precision: a fine-tuned retrieval view removed most unretrievable pairs, and competitor-aware
models (competition features, LLM reranker, owner model) attacked the name collisions that F0.5 punishes most. The
remaining dev loss is dominated by empty-address records whose name is shared by several entities, which the text
alone cannot resolve. The unseen country is the main gap to the leaderboard top; at production scale we would distil
the scorers into one cross-encoder, keep the ~4-candidate cascade, and route the ambiguous band to human review.
