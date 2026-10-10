"""Final pipeline diagram (PNG + SVG) for the approach doc and README.

    python scripts/plot_architecture.py                                   # E033 -> docs/architecture.png
    python scripts/plot_architecture.py --final E034 --owner --dev 0.9912 --out docs/architecture
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, LINE = "#1f2933", "#52606d", "#9aa5b1"
FILL = {"data": "#eef2f7", "block": "#e3f0fb", "score": "#fdf1e3", "model": "#e6f4ea", "out": "#f3e8fd", "side": "#fff8e1"}
EDGE = {"data": "#7b8794", "block": "#2f80c2", "score": "#d9822b", "model": "#2e8b57", "out": "#8e44ad", "side": "#c9a227"}

ap = argparse.ArgumentParser()
ap.add_argument("--final", default="E033")
ap.add_argument("--dev", default="0.99099", help="dev macro F0.5 of the final model (test-like density)")
ap.add_argument("--owner", action="store_true", help="include the OW04 listwise owner model (E034)")
ap.add_argument("--out", default="docs/architecture", help="output path without extension")
ap.add_argument("--france-unc", default="0.331", help="final French uncertain pairs per S1")
ap.add_argument("--st2", action="store_true", help="second France self-training round (E036)")
ap.add_argument("--rescue", action="store_true", help="unseen-country candidate rescue (E037)")
ap.add_argument("--cand-test", default="Test: 3.98 / S1 (US 3.76, India 3.86, France 4.92), 6.9M pairs")
A = ap.parse_args()

fig, ax = plt.subplots(figsize=(14, 16.8 if A.owner else 16))
ax.set_xlim(0, 100)
ax.set_ylim(-20, 124)
ax.axis("off")


def box(x, y, w, h, kind, title, lines, stat=None):
    """Rounded box centred at (x, y) with a bold title, body lines and an optional stat line; h grows to fit."""
    h = max(h, 6.6 + 2.55 * len(lines) + (2.55 if stat else 0))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x - w / 2 + 1.6, y + h / 2 - 2.2, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x - w / 2 + 1.6, y + h / 2 - 5.4 - i * 2.55, t, ha="left", va="top", fontsize=9.2, color=MUTED)
    if stat:
        ax.text(x - w / 2 + 1.6, y + h / 2 - 5.4 - len(lines) * 2.55, stat, ha="left", va="top", fontsize=9.6,
                fontweight="bold", color=EDGE[kind])
    return y - h / 2, y + h / 2


def arrow(x1, y1, x2, y2, label=None, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=LINE,
                                 linestyle=(0, (4, 3)) if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, label, fontsize=8.6, color=MUTED, va="center", style="italic")


CX, W = 37, 66
ax.text(50, 122.5, f"Business entity resolution: final pipeline ({A.final})", ha="center", fontsize=17, fontweight="bold",
        color=INK)
ax.text(50, 119.6, "Amazon ML Challenge 2026 - team SHSHSHSH - cascade from high recall to high precision",
        ha="center", fontsize=10.5, color=MUTED)

STEPS = [
    ("data", "Inputs: provided records only (name, address, country)",
     ["Source 1: 1.73M test entities (US, India, France)   |   Sources 2 + 3: ~10M records",
      "Normalisation: Unicode, anyascii transliteration (Indic scripts),",
      "hand-written legal-suffix and street-type lists"], None),
    ("block", "A. Dense retrieval (blocking index)",
     ["4 multilingual-e5 views: name, address, name+address, fine-tuned bi-encoder (folds 1-4)",
      "Same-country partition, top-10 per view per source",
      "Exact GPU k-NN here; FAISS IVF/HNSW sharded by country at billion scale"],
     "~62 candidates / S1   ·   pair recall 0.9976 (dev)"),
    ("block", "B. Stage-1 LightGBM pre-filter",
     ["32 cheap features: embedding cosines, rapidfuzz name/address similarity,",
      "number agreement, rank context; keeps the top 15 pairs per S1"], "15 / S1   ·   pair recall 0.9974 (dev)"),
    ("score", "Pair scorers (out-of-sample on the stage-2 fit pool)",
     ["Cross-encoders: e5-small (E008), e5-base (E016), e5-large (E027)",
      "e5-large adapted to France: francized train pairs (E029) + test self-training (E030)",
      "Qwen3-4B LoRA reranker on the uncertain band (prompt shows the competing S1s)"]
     + (["Listwise owner model OW04 (e5-large, folds 1-4): reads a contested record with up to",
         "6 competing S1s and predicts which one owns it -> owner probability + owner margin"] if A.owner else []),
     None),
    ("out", "C. Candidate filter  ->  candidate_pairs.tsv",
     ["stage-1 prob >= 0.5  OR  cross-encoder (E016) >= 0.05: exactly the matcher's input"]
     + (["France: also kept if the France-adapted e5-large CE >= 0.5 (rescue)"] if A.rescue else []) +
     [
      A.cand_test],
     "3.83 / S1 (dev)   ·   pair recall 0.991   ·   reduction ratio 99.99996%"),
    ("model", "Stage-2 LightGBM matcher (fusion)",
     (["Fuses every score above, incl. the two owner features (blank where uncontested),",
       "with cluster (sibling) features and full-population competition features"] if A.owner else
      ["All scores above + cluster (sibling) features + full-population competition"]) +
     (["(margin over each record's best other S1) at test-like density (keep 0.78)"] if A.owner else
      ["features (margin over each record's best other S1), test-like density (keep 0.78)"]) +
     ["Fitted on fold-0 S1s outside dev, unseen by every upstream learned model"], None),
    ("out", "Decision layer  ->  matching_results.tsv",
     ["Threshold 0.75 (0.85 for the country absent from train);",
      "each record to at most one S1; empty list when nothing passes"],
     f"macro F0.5 {A.dev} (dev, test-like density)"),
]
top, GAP, centre = 116.0, 3.6, {}
for i, (kind, title, lines, stat) in enumerate(STEPS):
    h = 6.6 + 2.55 * len(lines) + (2.55 if stat else 0)
    lo, hi = box(CX, top - h / 2, W, h, kind, title, lines, stat)
    centre[i] = (lo + hi) / 2
    if i + 1 < len(STEPS):
        arrow(CX, lo - 0.3, CX, lo - GAP + 0.6)
    top = lo - GAP
bottom = top + GAP
ax.set_ylim(bottom - 12, 124)

SX, SW = 88, 23
for idx, title, lines in ((3, "France (unseen in train)",
                           ["Francized synthetic pairs:", "hand-written word list", "(Street->Rue, LLC->SARL, ...)",
                            "Self-training on confident", "test predictions (allowed)"]
                           + (["Round 2: teacher = owner-model", "run, 450k French pseudo-labels"] if A.st2 else [])
                           + ["French uncertain pairs/S1:", f"0.447 -> {A.france_unc}", "Stricter threshold 0.85"]),
                          (5, "Validation",
                           ["Dev: 100k fold-0 S1s,", "scored at test-like density", "Every learned model: folds 1-4",
                            "Public-LB probes for France", "(threshold 0.55 / 0.75 / 0.90)",
                            "France: manual audit +", "drift guard on model updates",
                            "Validators: ours + organisers'"])):
    box(SX, centre[idx], SW, 0, "side", title, lines)
    arrow(SX - SW / 2 - 0.6, centre[idx], CX + W / 2 + 0.6, centre[idx], dashed=True)

legend = [("data", "input"), ("block", "blocking"), ("score", "pair scoring"), ("out", "deliverable"),
          ("model", "matcher"), ("side", "adaptation / validation")]
for i, (k, t) in enumerate(legend):
    lx = 8 + i * 15.2
    ax.add_patch(plt.Rectangle((lx, bottom - 5.6), 2.6, 2.2, fc=FILL[k], ec=EDGE[k], lw=1.3, zorder=5, clip_on=False))
    ax.text(lx + 3.4, bottom - 4.1, t, fontsize=9, va="center", color=MUTED)
ax.text(50, bottom - 8.5, "Compute: ~34 GPU-hours on one RTX 5080 (16 GB) + 64 GB RAM"
        + (" (+ OW04 trained on a teammate's GPU)" if A.owner else "")
        + ". Models: multilingual-e5 (MIT), Qwen3-4B (Apache-2.0), LightGBM (MIT).", ha="center", fontsize=9, color=MUTED)

out = Path(A.out)
fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
print(f"wrote {out}.png, {out}.svg")
