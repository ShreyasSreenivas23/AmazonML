"""Submission #1 probe: every test S1 entity with an empty match list.

    python scripts/make_probe_empty.py   # -> submissions/sub01_empty/{matching_results,candidate_pairs}.tsv

Its public-LB score equals the public singleton rate (empty prediction scores 1.0 only on true singletons),
and it proves the upload format end to end. Validated with our checker; also run the organisers' validator.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.er_data import dataset_dir, load  # noqa: E402
from src.er_submission import write_outputs  # noqa: E402


def main() -> None:
    t1, _, _, _ = load("test", with_gt=False)
    out = ROOT / "submissions" / "sub01_empty"
    write_outputs({}, {}, t1.entity_id, out, test_dir=dataset_dir() / "test")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
