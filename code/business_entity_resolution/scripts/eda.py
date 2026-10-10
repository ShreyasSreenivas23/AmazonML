"""P0 EDA for the entity-resolution task. Writes runs/eda/report.md (compact; safe to read in full).

    python scripts/eda.py            # needs the parquet cache: python -m src.er_data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import get_paths  # noqa: E402
from src.er_data import load  # noqa: E402

OUT = get_paths().runs / "eda"
lines: list[str] = []


def h(t: str) -> None:
    lines.append(f"\n## {t}\n")


def p(t: str = "") -> None:
    lines.append(t)


def table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> None:
    p(df.to_markdown(floatfmt=floatfmt.strip("{}:")) if hasattr(df, "to_markdown") else df.to_string())


def script_of(s: pd.Series) -> pd.Series:
    """Coarse script class per string."""
    out = np.full(len(s), "ascii", dtype=object)
    non_ascii = s.str.contains(r"[^\x00-\x7f]", regex=True).to_numpy()
    deva = s.str.contains(r"[ऀ-ॿ]", regex=True).to_numpy()
    latin_ext = s.str.contains(r"[À-ɏ]", regex=True).to_numpy()
    out[non_ascii] = "other_non_ascii"
    out[latin_ext] = "latin_accented"
    out[deva] = "devanagari"
    return pd.Series(out, index=s.index)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, gt = load("train")
    t1, t2, t3, _ = load("test", with_gt=False)
    p(f"# EDA report (generated {time.strftime('%Y-%m-%d %H:%M')})")

    # ---------------------------------------------------------------- sizes & fields
    h("1. Sizes, empties, countries")
    rows = []
    for split, frames in (("train", (s1, s2, s3)), ("test", (t1, t2, t3))):
        for k, f in enumerate(frames, 1):
            rows.append({"split": split, "src": f"S{k}", "rows": len(f),
                         "empty_name": (f.business_name.str.strip() == "").mean(),
                         "empty_addr": (f.business_address.str.strip() == "").mean(),
                         "dup_id": f.entity_id.duplicated().sum(),
                         "dup_name_addr": f.duplicated(["business_name", "business_address"]).mean(),
                         "countries": dict(f.country.value_counts(normalize=True).round(4))})
    table(pd.DataFrame(rows))
    p(f"\nID prefix check: all S1 ids start 'S1-': {s1.entity_id.str.startswith('S1-').all()}, "
      f"S2: {s2.entity_id.str.startswith('S2-').all()}, S3: {s3.entity_id.str.startswith('S3-').all()}")

    # ---------------------------------------------------------------- ground truth structure
    h("2. Ground truth structure")
    gt = gt.assign(src=gt.cand_id.str[:2])
    per = gt.groupby("s1_id").src.value_counts().unstack(fill_value=0)
    per = per.reindex(s1.entity_id, fill_value=0)
    n_tot = per.sum(1)
    p(f"- S1 entities: {len(s1):,}; match pairs: {len(gt):,} (S2 {int((gt.src == 'S2').sum()):,}, S3 {int((gt.src == 'S3').sum()):,})")
    p(f"- **Singleton rate (no matches): {(n_tot == 0).mean():.4f}**; only-S2: {((per.S2 > 0) & (per.S3 == 0)).mean():.4f}; "
      f"only-S3: {((per.S2 == 0) & (per.S3 > 0)).mean():.4f}; both: {((per.S2 > 0) & (per.S3 > 0)).mean():.4f}")
    p(f"- Matches per S1: mean {n_tot.mean():.3f}, median {n_tot.median():.0f}, p90 {n_tot.quantile(.9):.0f}, "
      f"p99 {n_tot.quantile(.99):.0f}, max {n_tot.max()}")
    p("- Distribution of total matches per S1:")
    table(n_tot.clip(upper=10).value_counts(normalize=True).sort_index().rename("share").to_frame())
    p("- S2 matches per S1 / S3 matches per S1 (share, clipped at 5):")
    table(pd.DataFrame({"S2": per.S2.clip(upper=5).value_counts(normalize=True).sort_index(),
                        "S3": per.S3.clip(upper=5).value_counts(normalize=True).sort_index()}))
    multi = gt.cand_id.value_counts()
    p(f"- **Records matched to >1 S1: {(multi > 1).sum():,}** (max S1 per record {multi.max()})")
    p(f"- Unknown ids in GT (not in S2/S3): {(~gt.cand_id.isin(pd.concat([s2.entity_id, s3.entity_id]))).sum():,}")
    p(f"- Share of S2 records matched: {s2.entity_id.isin(gt.cand_id).mean():.4f}; S3: {s3.entity_id.isin(gt.cand_id).mean():.4f} "
      "(the rest are distractors with no S1 counterpart)")

    # ---------------------------------------------------------------- countries
    h("3. Countries")
    c1 = s1.set_index("entity_id").country
    cc = pd.concat([s2.set_index("entity_id").country, s3.set_index("entity_id").country])
    pair_c = pd.DataFrame({"c1": c1.reindex(gt.s1_id).to_numpy(), "c2": cc.reindex(gt.cand_id).to_numpy()})
    p(f"- **Cross-country match pairs: {(pair_c.c1 != pair_c.c2).mean():.5f}**")
    table(pd.crosstab(pair_c.c1, pair_c.c2))
    p(f"- Singleton rate by S1 country: {dict(pd.Series(n_tot.to_numpy() == 0, index=s1.country.to_numpy()).groupby(level=0).mean().round(4))}")
    p(f"- Test S1 country share: {dict(t1.country.value_counts(normalize=True).round(4))}")
    p(f"- (S2+S3)/S1 ratio: train {(len(s2) + len(s3)) / len(s1):.3f}, test {(len(t2) + len(t3)) / len(t1):.3f}")
    for name, f in (("test S1", t1), ("test S2", t2), ("test S3", t3)):
        vc = f.country.value_counts()
        p(f"- {name} countries: {dict(vc)}")

    # ---------------------------------------------------------------- scripts
    h("4. Scripts in names")
    rows = []
    for split, frames in (("train", (s1, s2, s3)), ("test", (t1, t2, t3))):
        for k, f in enumerate(frames, 1):
            sc = script_of(f.business_name)
            for c, g in sc.groupby(f.country.to_numpy()):
                rows.append({"split": split, "src": f"S{k}", "country": c, **g.value_counts(normalize=True).round(4).to_dict()})
    table(pd.DataFrame(rows).fillna(0))

    # ---------------------------------------------------------------- similarity of true pairs
    h("5. How similar are true pairs? (sample of 200k pairs)")
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.fuzz import token_set_ratio

    samp = gt.sample(min(200_000, len(gt)), random_state=0)
    n1 = s1.set_index("entity_id").reindex(samp.s1_id)
    oth = pd.concat([s2, s3]).set_index("entity_id").reindex(samp.cand_id)
    a, b = n1.business_name.str.lower().to_numpy(), oth.business_name.str.lower().to_numpy()
    tsr = np.array([token_set_ratio(x, y) for x, y in zip(a, b)])
    jw = np.array([JaroWinkler.similarity(x, y) for x, y in zip(a, b)])
    exact = (a == b)
    p(f"- exact lowercase name equality: {exact.mean():.4f}")
    p(f"- token_set_ratio: p10 {np.percentile(tsr, 10):.0f}, p25 {np.percentile(tsr, 25):.0f}, median {np.median(tsr):.0f}")
    p(f"- Jaro-Winkler: p10 {np.percentile(jw, 10):.3f}, median {np.median(jw):.3f}")
    by_src = pd.DataFrame({"src": samp.cand_id.str[:2].to_numpy(), "tsr": tsr, "country": n1.country.to_numpy()})
    table(by_src.groupby(["country", "src"]).tsr.describe()[["mean", "25%", "50%"]])
    # negatives for contrast: sampled S1 names vs RANDOM S2/S3 records of the SAME country (matched or not)
    rng = np.random.default_rng(0)
    pool = pd.concat([s2, s3])[["business_name", "country"]]
    neg_a, neg_b = [], []
    for c, g in n1.head(50_000).groupby("country"):
        cand = pool.business_name[pool.country == c].to_numpy()
        neg_a += g.business_name.str.lower().tolist()
        neg_b += [x.lower() for x in cand[rng.integers(0, len(cand), size=len(g))]]
    tsr_neg = np.array([token_set_ratio(x, y) for x, y in zip(neg_a, neg_b)])
    p(f"- random same-country negatives token_set_ratio: median {np.median(tsr_neg):.0f}, p90 {np.percentile(tsr_neg, 90):.0f}")

    # ---------------------------------------------------------------- addresses
    h("6. Addresses")
    for name, f in (("S1", s1), ("S2", s2), ("S3", s3), ("test S1", t1)):
        ad = f.business_address
        for c in sorted(f.country.unique()):
            m = f.country == c
            x = ad[m]
            empty = (x.str.strip() == "").mean()
            d5 = x.str.contains(r"(?<![0-9])[0-9]{5}(?![0-9])").mean()
            d6 = x.str.contains(r"(?<![0-9])[0-9]{6}(?![0-9])").mean()
            near = x.str.contains(r"(?i)\bnear\b").mean()
            commas = x.str.count(",").mean()
            p(f"- {name}/{c}: empty {empty:.3f} | has 5-digit {d5:.3f} | has 6-digit {d6:.3f} | has 'near' {near:.3f} "
              f"| mean commas {commas:.2f}")

    # ---------------------------------------------------------------- structure / leakage checks
    h("7. Structure checks (to protect validation, never to exploit)")
    pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    pos_o = pd.concat([pd.Series(np.arange(len(s2)), index=s2.entity_id), pd.Series(np.arange(len(s3)), index=s3.entity_id)])
    x, y = pos1.reindex(gt.s1_id).to_numpy(), pos_o.reindex(gt.cand_id).to_numpy()
    p(f"- Spearman(row position S1, row position of match): {pd.Series(x).corr(pd.Series(y), method='spearman'):.4f}")
    num1 = gt.s1_id.str[3:].astype("int64").to_numpy()
    num2 = gt.cand_id.str[3:].astype("int64").to_numpy()
    p(f"- Spearman(S1 id number, match id number): {pd.Series(num1).corr(pd.Series(num2), method='spearman'):.4f}")
    p(f"- ID number lengths S1: {dict(s1.entity_id.str.len().value_counts().sort_index().head(8))}")
    ids_all = pd.concat([s1.entity_id.str[3:], s2.entity_id.str[3:], s3.entity_id.str[3:], t1.entity_id.str[3:]])
    p(f"- ID numbers shared across sources/splits (numeric part): {ids_all.duplicated().sum():,}")
    tr_names = set(s1.business_name)
    p(f"- Test S1 names that also appear exactly as a train S1 name: {t1.business_name.isin(tr_names).mean():.4f}")

    # ---------------------------------------------------------------- examples
    h("8. Examples of true pairs (random 25)")
    ex = samp.head(25)
    for s1_id, cand in zip(ex.s1_id, ex.cand_id):
        r1, r2 = n1.loc[s1_id], oth.loc[cand]
        r1 = r1.iloc[0] if isinstance(r1, pd.DataFrame) else r1
        r2 = r2.iloc[0] if isinstance(r2, pd.DataFrame) else r2
        p(f"- `{r1.business_name}` | `{r1.business_address}` ({r1.country})  →  {cand[:2]}: `{r2.business_name}` | `{r2.business_address}` ({r2.country})")
    h("9. Examples of test France S1 records (15)")
    for _, r in t1[t1.country == "France"].sample(15, random_state=0).iterrows():
        p(f"- `{r.business_name}` | `{r.business_address}`")
    h("10. Examples of Devanagari names in train (10, with their S1)")
    deva = gt[gt.cand_id.isin(pd.concat([s2, s3]).loc[lambda d: d.business_name.str.contains(r"[ऀ-ॿ]"), "entity_id"])].head(10)
    for s1_id, cand in zip(deva.s1_id, deva.cand_id):
        p(f"- S1 `{s1.set_index('entity_id').loc[s1_id].business_name}`  →  `{pd.concat([s2, s3]).set_index('entity_id').loc[cand].business_name}`")

    p(f"\n_EDA runtime {time.time() - t0:.0f}s_")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
