"""Country probe: copy a submission and empty every matched list for one country (leaderboard diagnostic).

    python scripts/make_country_probe.py submissions/sub03_E009 France submissions/probe_france_sub03

F_country on the public set = (LB_full - LB_probe) / share_country + no_match_rate_country
(emptied rows score 1.0 only on true no-match entities). Candidates are copied unchanged.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.er_data import dataset_dir, load  # noqa: E402
from src.er_submission import validate_outputs  # noqa: E402


def main() -> int:
    src, country, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    t1, _, _, _ = load("test", with_gt=False)
    target = set(t1.entity_id[t1.country == country])
    if not target:
        print(f"no test S1 with country {country!r}")
        return 1
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(src / "matching_results.tsv", encoding="utf-8") as fi, \
            open(dst / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fo:
        fo.write(fi.readline())
        for line in fi:
            s1 = line.split("\t", 1)[0]
            if s1 in target:
                fo.write(f"{s1}\t\n")
                n += 1
            else:
                fo.write(line)
    shutil.copyfile(src / "candidate_pairs.tsv", dst / "candidate_pairs.tsv")
    rep = validate_outputs(dst / "matching_results.tsv", dst / "candidate_pairs.tsv", test_dir=dataset_dir() / "test")
    print(rep)
    print(f"emptied {n:,} {country} rows ({n / len(t1):.4f} of test S1)")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
