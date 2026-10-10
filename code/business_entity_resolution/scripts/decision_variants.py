"""Write submission variants from ONE stage-2 run's saved test probabilities (no rebuild).

Every variant shares that run's candidate set (candidate_pairs.tsv is byte-identical across variants); only the decision
threshold differs: `--t-seen` for S1 countries present in train, `--t-unseen` for countries absent from train (read
from the data, never listed by name). Used for the final-day threshold sweep on the public LB (France has no labels).

    python scripts/decision_variants.py --probs runs/E015/sub_E023a_core/test_probs_stage2.parquet \
        --out submissions/sub_E023a_t75_u90 --t-seen 0.75 --t-unseen 0.90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.er_data import cache_dir, dataset_dir, load  # noqa: E402
from src.er_decide import decide  # noqa: E402
from src.er_submission import write_outputs  # noqa: E402


def main() -> None:
    """Decide matches per S1 with country-group thresholds and write both TSVs (validated)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="test_probs_stage2.parquet of the chosen run (s1_id, cand_id, prob)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-seen", type=float, required=True)
    ap.add_argument("--t-unseen", type=float, required=True)
    a = ap.parse_args()
    probs = pd.read_parquet(a.probs)
    s1 = load("test", with_gt=False)[0]
    train_countries = set(pd.read_parquet(cache_dir() / "train_s1.parquet", columns=["country"]).country.unique())
    ctry = s1.set_index("entity_id").country
    unseen = ~ctry.reindex(probs.s1_id).isin(train_countries).to_numpy()
    matches = decide(probs[~unseen], a.t_seen, assign=True)
    matches.update(decide(probs[unseen], a.t_unseen, assign=True))
    cands: dict[str, list[str]] = {}
    for s, c in zip(probs.s1_id.to_numpy(), probs.cand_id.to_numpy()):
        cands.setdefault(s, []).append(c)
    write_outputs(matches, cands, s1.entity_id.to_numpy(), Path(a.out), test_dir=dataset_dir() / "test")
    print(f"unseen-country S1 pairs: {int(unseen.sum()):,} | matched S1s: {len(matches):,}")


if __name__ == "__main__":
    main()
