"""Build the final package in the exact layout the organisers require (docs/COMPETITION.md):

    <team>_submission.zip
    ├── output/matching_results.tsv, output/candidate_pairs.tsv
    ├── code/business_entity_resolution/{src/, configs/, scripts/, monitor/, README.md, requirements.txt, ...}
    └── Documentation_template.md   (our filled-in methodology; .pdf also accepted)

    python scripts/make_submission_zip.py --team <team_name> --outputs output/ \
        --doc docs/Documentation_template.md --test-dir data/dataset/test

- code comes from git-tracked files only (no data, weights, .env); refuses uncommitted changes unless --allow-dirty
- code/.../README.md is docs/REPRODUCE.md (exact end-to-end run instructions), not the repo README
- both TSVs are validated with src.er_submission (incl. matches ⊆ candidates) before zipping

Per-submission code upload (the portal asks for a code zip with EVERY leaderboard upload):
    python scripts/make_submission_zip.py --code-only --ref sub-03
    -> dist/code_sub-03.zip: code/business_entity_resolution/... exactly as committed at that tag
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CODE_PREFIX = "code/business_entity_resolution"
CODE_DIRS = ("src/", "configs/", "scripts/", "monitor/", "tests/")
CODE_FILES = ("requirements.txt", "requirements-core.txt", "pyproject.toml", ".env.example")
MAX_FILE_MB = 20
# Dev-only tooling that makes network calls (image downloader, Discord alerts). Not used by the pipeline; kept out
# of submission zips so reviewers see a pipeline with zero network access (organiser rule: no external data).
EXCLUDE = {"src/images.py", "tests/test_images.py", "monitor/notify.py", "monitor/watch.py", "monitor/demo.py",
           "tests/test_monitor.py", "scripts/prefetch_models.py"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def git_bytes(ref: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def code_only(ref: str, out_dir: str) -> int:
    """Zip the code as it exists at `ref` (a sub-NN tag), in the organisers' code/ layout."""
    sha = git("rev-parse", "--short", f"{ref}^{{commit}}")
    files = [f for f in git("ls-tree", "-r", "--name-only", ref).splitlines()
             if (f.startswith(CODE_DIRS) or f in CODE_FILES) and f not in EXCLUDE]
    if any(Path(f).name == ".env" for f in files):
        print("ERROR: .env tracked at this ref")
        return 1
    out = ROOT / out_dir / f"code_{ref}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    readme = git_bytes(ref, "docs/REPRODUCE.md") if "docs/REPRODUCE.md" in git("ls-tree", "-r", "--name-only", ref) else b""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f"{CODE_PREFIX}/{f}", git_bytes(ref, f))
        if readme:
            z.writestr(f"{CODE_PREFIX}/README.md", readme)
        info = [f"ref: {ref}", f"commit: {git('rev-parse', f'{ref}^{{commit}}')}", f"built: {datetime.now(timezone.utc).isoformat()}"]
        z.writestr(f"{CODE_PREFIX}/BUILD_INFO.txt", "\n".join(info) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MB, {len(files)} code files, {ref} = {sha})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-only", action="store_true", help="just the code zip for a leaderboard upload")
    ap.add_argument("--ref", default="HEAD", help="git ref the code is taken from (use the sub-NN tag)")
    ap.add_argument("--team", help="team name used in <team>_submission.zip (full package)")
    ap.add_argument("--outputs", default="output", help="folder with matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--doc", help="filled-in Documentation_template.md (or .pdf); full package only")
    ap.add_argument("--readme", default="docs/REPRODUCE.md", help="becomes code/.../README.md")
    ap.add_argument("--test-dir", help="test split dir for full validation (strongly recommended)")
    ap.add_argument("--no-outputs", action="store_true",
                    help="leave output/*.tsv out of the zip (small upload; the TSV goes in its own portal field)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--out-dir", default="dist")
    a = ap.parse_args()
    if a.code_only:
        return code_only(a.ref, a.out_dir)
    if not a.team or not a.doc:
        ap.error("--team and --doc are required for the full package (or use --code-only)")

    if a.ref == "HEAD" and git("status", "--porcelain", "--untracked-files=no") and not a.allow_dirty:
        print("ERROR: uncommitted changes. Commit first (the zip must be reproducible from a commit).")
        return 1
    outputs = Path(a.outputs)
    match, cand = outputs / "matching_results.tsv", outputs / "candidate_pairs.tsv"
    from src.er_submission import validate_outputs

    rep = validate_outputs(match, cand, test_dir=a.test_dir)
    print(rep)
    if not rep.ok:
        return 1
    if not Path(a.doc).exists():
        print(f"ERROR: missing {a.doc}")
        return 1

    # code (and its README) come from the git ref, so the zip matches the tagged submission exactly
    tracked = git("ls-tree", "-r", "--name-only", a.ref).splitlines()
    files = [f for f in tracked if (f.startswith(CODE_DIRS) or f in CODE_FILES) and f not in EXCLUDE]
    if any(Path(f).name == ".env" for f in files):
        print("ERROR: .env is tracked at this ref")
        return 1
    blobs = {f: git_bytes(a.ref, f) for f in files}
    big = [f for f, b in blobs.items() if len(b) > MAX_FILE_MB * 2**20]
    if big:
        print(f"ERROR: refusing to zip large files: {big}")
        return 1
    if a.readme not in tracked:
        print(f"ERROR: {a.readme} not committed at {a.ref}")
        return 1

    sha = git("rev-parse", "--short", f"{a.ref}^{{commit}}")
    out = ROOT / a.out_dir / (f"{a.team}_submission" + ("_code" if a.no_outputs else "") + ".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    doc_name = "Documentation_template" + Path(a.doc).suffix
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        if not a.no_outputs:
            z.write(match, "output/matching_results.tsv")
            z.write(cand, "output/candidate_pairs.tsv")
        for f, b in blobs.items():
            z.writestr(f"{CODE_PREFIX}/{f}", b)
        z.writestr(f"{CODE_PREFIX}/README.md", git_bytes(a.ref, a.readme))
        # the zip root holds exactly the spec's three items: the diagram the doc embeds goes inside the code folder,
        # and the doc's relative image link is rewritten to that location
        arch = Path(a.doc).parent / "architecture.png"
        doc_bytes = Path(a.doc).read_bytes()
        if arch.exists():
            z.write(arch, f"{CODE_PREFIX}/docs/architecture.png")
            doc_bytes = doc_bytes.replace(b"](architecture.png)", f"]({CODE_PREFIX}/docs/architecture.png)".encode())
        z.writestr(doc_name, doc_bytes)
        summary = Path(a.doc).parent / "Approach_Summary.md"      # the guidelines' 1-2 page approach document
        if summary.exists():
            z.write(summary, f"{CODE_PREFIX}/docs/Approach_Summary.md")
        info = [f"ref: {a.ref}", f"commit: {git('rev-parse', f'{a.ref}^{{commit}}')}",
                f"built: {datetime.now(timezone.utc).isoformat()}"]
        z.writestr(f"{CODE_PREFIX}/BUILD_INFO.txt", "\n".join(info) + "\n")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    required = [f"{CODE_PREFIX}/README.md", f"{CODE_PREFIX}/requirements.txt", doc_name]
    if not a.no_outputs:
        required += ["output/matching_results.tsv", "output/candidate_pairs.tsv"]
    missing = [r for r in required if r not in names] + ([] if any(n.startswith(f"{CODE_PREFIX}/src/") for n in names)
                                                         else [f"{CODE_PREFIX}/src/"])
    if missing:
        print(f"ERROR: zip is missing {missing}")
        return 1
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MB, {len(names)} entries, commit {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
