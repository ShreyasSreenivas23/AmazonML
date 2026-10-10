"""Pre-upload gate: validate the file, require a clean tree, tag the commit sub-NN, log a ledger row.

    python scripts/tag_submission.py submissions/sub02_E003 --exp E003-baseline --cv 0.8123
    (folder with matching_results.tsv + candidate_pairs.tsv; validated against data/dataset/test)

Steps: validate predictions -> refuse if uncommitted changes -> create annotated tag sub-NN
on HEAD -> append a row to docs/SUBMISSIONS.md (LB column left blank) -> print the git
commands to commit the ledger and push the tag. Upload manually on Unstop, then fill LB.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "docs" / "SUBMISSIONS.md"
IST = timezone(timedelta(hours=5, minutes=30))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="folder containing matching_results.tsv and candidate_pairs.tsv")
    ap.add_argument("--exp", required=True, help="experiment id, e.g. E012-lgbm-text")
    ap.add_argument("--cv", required=True, help="local CV score")
    ap.add_argument("--test-dir", default=None, help="default: <DATA_DIR>/dataset/test")
    ap.add_argument("--remaining", default="?", help="submissions remaining today AFTER this one")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from src.er_data import dataset_dir
    from src.er_submission import validate_outputs

    folder = Path(a.predictions)
    rep = validate_outputs(folder / "matching_results.tsv", folder / "candidate_pairs.tsv",
                           test_dir=a.test_dir or dataset_dir() / "test")
    print(rep)
    if not rep.ok:
        return 1
    if git("status", "--porcelain", "--untracked-files=no"):
        print("ERROR: commit code+config first; the tag must point at exactly what produced this file.")
        return 1
    existing = [int(m.group(1)) for t in git("tag", "--list", "sub-*").splitlines() if (m := re.match(r"sub-(\d+)$", t))]
    n = max(existing, default=0) + 1
    tag = f"sub-{n:02d}"
    ts = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    row = f"| {n} | {ts} | {a.exp} | `{tag}` | {a.cv} |  | {a.remaining} | {Path(a.predictions).name} |\n"
    if a.dry_run:
        print("would tag", tag, "and append:", row)
        return 0
    git("tag", "-a", tag, "-m", f"submission {n}: {a.exp} CV={a.cv}")
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(row)
    print(f"tagged {tag} -> {git('rev-parse', '--short', 'HEAD')}; ledger row appended.\nNext:\n"
          f"  git add docs/SUBMISSIONS.md && git commit -m \"docs: log {tag} ({a.exp}, CV {a.cv})\"\n"
          f"  git push origin HEAD {tag}\n  upload on Unstop, then fill the Public LB column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
