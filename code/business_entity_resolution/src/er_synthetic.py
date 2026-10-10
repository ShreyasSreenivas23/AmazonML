"""Synthetic entity-resolution data in the EXACT organiser format (docs/COMPETITION.md).

Lets any worker (a cloud session with no real data, CI, tests) run the full pipeline end to end:

    python -m src.er_synthetic --out data_synth --n 400
    # -> data_synth/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
    #    data_synth/dataset/test/test_source{1,2,3}.tsv  (+ test_ground_truth.tsv for scoring)

Mimics the statement's noise: abbreviations, legal-suffix swaps, '&' vs 'and', word swaps, typos, missing
address parts, landmark references, singletons, and distractor records. Train has US + India only;
test adds France (the unseen-country shift). It is NOT a model of the real data: use it for correctness,
never for tuning.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

WORDS = ["Sharma", "Global", "Sunrise", "Metro", "Lakshmi", "Pioneer", "Blue", "River", "Summit", "Ganesh",
         "Atlas", "Crown", "Eagle", "Harbor", "Lotus", "Maple", "Nova", "Orchid", "Prime", "Royal", "Silver"]
KIND = ["Traders", "Motors", "Foods", "Textiles", "Pharma", "Logistics", "Electronics", "Bakery", "Consulting"]
COUNTRY = {
    "US": {"suffix": [("Corporation", "Corp"), ("Incorporated", "Inc"), ("Limited Liability Company", "LLC")],
           "street": [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd")],
           "cities": [("Springfield", "IL"), ("Austin", "TX"), ("Denver", "CO")], "zip": 5},
    "India": {"suffix": [("Private Limited", "Pvt Ltd"), ("Limited", "Ltd"), ("Limited Liability Partnership", "LLP")],
              "street": [("Road", "Rd"), ("Marg", "Mg"), ("Nagar", "Ngr")],
              "cities": [("Pune", "MH"), ("Chennai", "TN"), ("Jaipur", "RJ")], "zip": 6},
    "France": {"suffix": [("Société à responsabilité limitée", "SARL"), ("Société par actions simplifiée", "SAS")],
               "street": [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd")],
               "cities": [("Paris", "IDF"), ("Lyon", "ARA"), ("Marseille", "PACA")], "zip": 5},
}
HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def _typo(s: str, rng: random.Random) -> str:
    """Inject one random character-level typo."""
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:] if rng.random() < 0.5 else s[:i] + s[i + 1:]


def _entity(country: str, rng: random.Random) -> dict:
    """Draw a synthetic business (name, address, country)."""
    c = COUNTRY[country]
    full, abbr = rng.choice(c["suffix"])
    street_full, street_abbr = rng.choice(c["street"])
    city, state = rng.choice(c["cities"])
    words = rng.sample(WORDS, 2) + [rng.choice(KIND)]
    return {"core": words, "suffix": (full, abbr), "num": str(rng.randint(1, 999)),
            "street": (rng.choice(WORDS), street_full, street_abbr), "city": city, "state": state,
            "zip": "".join(rng.choice("0123456789") for _ in range(c["zip"])), "country": country}


def _render(e: dict, rng: random.Random, noisy: bool) -> tuple[str, str]:
    """Render a noisy copy of an entity as a source record."""
    core = list(e["core"])
    suffix = e["suffix"][1] if noisy and rng.random() < 0.5 else e["suffix"][0]
    if noisy:
        if rng.random() < 0.3:
            core[0], core[1] = core[1], core[0]
        if rng.random() < 0.3:
            core = [_typo(w, rng) if rng.random() < 0.5 else w for w in core]
        if rng.random() < 0.2:
            core.insert(1, "&" if rng.random() < 0.5 else "and")
        if rng.random() < 0.2:
            suffix = ""
    name = " ".join(core + ([suffix] if suffix else []))
    sname, sfull, sabbr = e["street"]
    parts = [f"{e['num']} {sname} {sabbr if noisy and rng.random() < 0.5 else sfull}", e["city"], e["state"], e["zip"]]
    if e["country"] == "France":
        parts = [f"{e['num']} {sabbr if noisy and rng.random() < 0.5 else sfull} {sname}", f"{e['zip']} {e['city']}"]
    if noisy:
        parts = [p for i, p in enumerate(parts) if i == 0 or rng.random() > 0.25]  # missing components
        if e["country"] == "India" and rng.random() < 0.3:
            parts.insert(1, f"Near {rng.choice(['SBI ATM', 'Bus Stand', 'City Mall'])}")
        if rng.random() < 0.15:
            parts.reverse()
    return name, ", ".join(parts)


def make_split(countries: list[str], n: int, rng: random.Random, prefix: str) -> tuple[list, list, list, dict]:
    """Build synthetic S1/S2/S3 sources and ground truth with the real schema."""
    s1, s2, s3, gt = [], [], [], {}
    c2 = c3 = 0
    for i in range(1, n + 1):
        e = _entity(rng.choice(countries), rng)
        sid = f"S1-{i:05d}"
        s1.append((sid, *_render(e, rng, noisy=False), e["country"]))
        matches = []
        r = rng.random()
        n2 = 0 if r < 0.3 else (1 if r < 0.85 else 2)    # ~30% singletons w.r.t. S2
        n3 = 0 if rng.random() < 0.4 else 1
        for _ in range(n2):
            c2 += 1
            s2.append((f"S2-{c2:05d}", *_render(e, rng, noisy=True), e["country"]))
            matches.append(f"S2-{c2:05d}")
        for _ in range(n3):
            c3 += 1
            s3.append((f"S3-{c3:05d}", *_render(e, rng, noisy=True), e["country"]))
            matches.append(f"S3-{c3:05d}")
        gt[sid] = matches
    for _ in range(n // 5):  # distractors: records with no S1 counterpart
        e = _entity(rng.choice(countries), rng)
        if rng.random() < 0.5:
            c2 += 1
            s2.append((f"S2-{c2:05d}", *_render(e, rng, noisy=True), e["country"]))
        else:
            c3 += 1
            s3.append((f"S3-{c3:05d}", *_render(e, rng, noisy=True), e["country"]))
    rng.shuffle(s2)
    rng.shuffle(s3)
    return s1, s2, s3, gt


def write_split(out: Path, split: str, s1, s2, s3, gt) -> None:
    """Write a synthetic split as organiser-format TSVs."""
    d = out / "dataset" / split
    d.mkdir(parents=True, exist_ok=True)
    for k, rows in ((1, s1), (2, s2), (3, s3)):
        with open(d / f"{split}_source{k}.tsv", "w", encoding="utf-8", newline="\n") as f:
            f.write(HEADER + "".join("\t".join(r) + "\n" for r in rows))
    with open(d / f"{split}_ground_truth.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n" + "".join(f"{k}\t{','.join(v)}\n" for k, v in gt.items()))


def generate(out: str | Path, n: int = 400, seed: int = 0) -> Path:
    """Create a small synthetic dataset for tests (no real data needed)."""
    out = Path(out)
    rng = random.Random(seed)
    write_split(out, "train", *make_split(["US", "India"], n, rng, "train"))
    write_split(out, "test", *make_split(["US", "India", "France"], max(n // 2, 10), rng, "test"))
    return out


def main() -> None:
    """CLI: write a synthetic dataset to a folder."""
    ap = argparse.ArgumentParser(prog="python -m src.er_synthetic")
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--n", type=int, default=400, help="train S1 entities (test gets n/2)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"wrote synthetic ER data to {generate(a.out, a.n, a.seed)}")


if __name__ == "__main__":
    main()
