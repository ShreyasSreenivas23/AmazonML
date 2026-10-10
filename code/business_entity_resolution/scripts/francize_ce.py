"""E029: continue the E027 e5-large cross-encoder on "francized" TRAIN pairs so it sees French-style vocabulary and noise
with real labels (France has no labels). Only provided training data + a small hand-written generic-word dictionary
(street types, legal forms, common words; no places, no entities, no geography).

    python scripts/francize_ce.py train      # ~35 min GPU: 300k francized + 150k original pairs, lr 1e-5
    python scripts/francize_ce.py check      # francized-dev + dev AUC/logloss, E027 vs E029 (labels from fold-0 dev)
    python scripts/francize_ce.py score      # re-score ONLY unseen-country test candidate rows -> runs/E029-ce
"""
from __future__ import annotations

import re
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import er_crossenc as ce  # noqa: E402
from src.er_data import cache_dir  # noqa: E402

BASE, OUT, CE_OUT = Path("runs/E027-ce-large/model"), Path("runs/E029-ce-fr/model"), Path("runs/E029-ce")
WORDS = {  # generic English/Indian-English words -> French equivalents (hand-written, no geography)
    "street": "Rue", "st": "Rue", "road": "Route", "rd": "Route", "avenue": "Avenue", "ave": "Avenue",
    "boulevard": "Boulevard", "blvd": "Boulevard", "lane": "Allée", "ln": "Allée", "drive": "Chemin", "dr": "Chemin",
    "place": "Place", "pl": "Place", "court": "Cour", "ct": "Cour", "square": "Place", "marg": "Rue", "way": "Voie",
    "inc": "SA", "incorporated": "SA", "llc": "SARL", "corp": "SAS", "corporation": "SAS", "ltd": "SAS",
    "limited": "SAS", "llp": "EURL", "pvt": "", "private": "", "co": "Cie", "company": "Compagnie", "and": "et",
    "sons": "Fils", "brothers": "Frères", "partners": "Associés", "group": "Groupe", "enterprises": "Entreprises",
    "trading": "Commerce", "center": "Centre", "centre": "Centre", "medical": "Médical", "school": "École",
    "consulting": "Conseil", "cafe": "Café", "bakery": "Boulangerie", "pharmacy": "Pharmacie", "hotel": "Hôtel",
    "services": "Services", "industries": "Industries", "association": "Association", "club": "Club",
    "sports": "Sport", "sport": "Sport", "society": "Société", "foundation": "Fondation", "clinic": "Clinique",
    "unit": "Bât", "suite": "Bât", "floor": "Étage", "near": "près de", "opposite": "face à",
}
ABBR = {"Rue": "R", "Avenue": "Av", "Boulevard": "Bd", "Allée": "All", "Chemin": "Ch", "Route": "Rte"}
TOK = re.compile(r"[A-Za-z]+|[^A-Za-z]+")


def _strip(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def francize(text: str, rng: np.random.Generator) -> str:
    """Map generic words to French, then add French-style surface noise (per record, random)."""
    if not isinstance(text, str) or not text:
        return text
    out = []
    for t in TOK.findall(text):
        k = t.lower()
        out.append(WORDS[k] if k in WORDS else t)
    s = re.sub(r"\s{2,}", " ", "".join(out)).strip()
    if rng.random() < 0.3:
        s = re.sub(r"\b(" + "|".join(ABBR) + r")\b", lambda m: ABBR[m.group(1)], s)
    if rng.random() < 0.3:
        s = s.upper()
    if rng.random() < 0.25:
        s = _strip(s)
    if rng.random() < 0.2:
        s = s.replace("'", "")
    return s


def francize_frame(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in ("business_name", "business_address"):
        out[col] = [francize(x, rng) for x in df[col].to_numpy()]
    out.index = out.index.astype(str) + "#fr"
    return out


def fr_pairs(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame, seed: int):
    """Francized copies of pairs + the francized left/right rows they need (ids suffixed with #fr)."""
    lf = francize_frame(left.loc[pairs.s1_id.unique()], seed)
    rf = francize_frame(right.loc[pairs.cand_id.unique()], seed + 1)
    p = pairs.assign(s1_id=pairs.s1_id + "#fr", cand_id=pairs.cand_id + "#fr")
    return p, lf, rf


def cmd_train() -> None:
    pairs = pd.read_parquet(BASE / "train_pairs.parquet")
    rng = np.random.default_rng(29)
    sub = pairs.iloc[rng.permutation(len(pairs))[:450_000]].reset_index(drop=True)
    left, right = ce.load_sources("train")
    fp, lf, rf = fr_pairs(sub.iloc[:300_000], left, right, 29)
    mix = pd.concat([fp, sub.iloc[300_000:]], ignore_index=True).sample(frac=1.0, random_state=29).reset_index(drop=True)
    L = pd.concat([left.loc[sub.s1_id.iloc[300_000:].unique()], lf])
    R = pd.concat([right.loc[sub.cand_id.iloc[300_000:].unique()], rf])
    print(f"E029 pairs {len(mix):,} ({(mix.s1_id.str.endswith('#fr')).mean():.0%} francized, {mix.y.mean():.1%} pos)", flush=True)
    for i in range(3):
        r = fp.iloc[i]
        print("  ", lf.loc[r.s1_id].tolist(), "||", rf.loc[r.cand_id].tolist(), "y", r.y, flush=True)
    ce.train(mix, L, R, OUT, model_name=str(BASE), epochs=1, batch=32, lr=1e-5, ckpt_every=2000, seed=29)


def cmd_check() -> None:
    from sklearn.metrics import log_loss, roc_auc_score
    folds = pd.read_parquet(cache_dir() / "folds_s1_k5.parquet")
    dev_ids = set(folds.loc[folds.dev, "s1_id"])
    dev = pd.read_parquet("runs/E015/dev_stage2_E023b_full.parquet", columns=["s1_id", "cand_id", "y"])
    dev = dev.sample(80_000, random_state=0).reset_index(drop=True)
    left, right = ce.load_sources("train")
    fp, lf, rf = fr_pairs(dev, left, right, 7)
    for name, d in (("E027", BASE), ("E029", OUT)):
        m, tok, dv = ce.load_model(d)
        for tag, P, L, R in (("dev", dev, left, right), ("francized-dev", fp, lf, rf)):
            s = ce.score_loaded(m, tok, dv, P, L, R, batch=256)
            print(f"{name} {tag}: AUC {roc_auc_score(P.y, s):.5f} logloss {log_loss(P.y, np.clip(s, 1e-6, 1 - 1e-6)):.5f}", flush=True)
        del m


def cmd_score() -> None:
    """E029 test_ce = E027 test_ce with unseen-country candidate rows re-scored; train_ce = copy of E027's."""
    train_countries = set(pd.read_parquet(cache_dir() / "train_s1.parquet", columns=["country"]).country.unique())
    ts1 = pd.read_parquet(cache_dir() / "test_s1.parquet", columns=["entity_id", "country"])
    unseen = set(ts1.loc[~ts1.country.isin(train_countries), "entity_id"])
    shutil.copytree("runs/E027-ce/train_ce", CE_OUT / "train_ce", dirs_exist_ok=True)
    (CE_OUT / "test_ce").mkdir(parents=True, exist_ok=True)
    left, right = ce.load_sources("test")
    m, tok, dv = ce.load_model(OUT)
    for f in sorted(Path("runs/E027-ce/test_ce").glob("*.parquet")):
        if (CE_OUT / "test_ce" / f.name).exists():
            continue
        c = pd.read_parquet(f)
        sel = (c.s1_id.isin(unseen) & c.ce_score.notna()).to_numpy()
        if sel.any():
            c.loc[sel, "ce_score"] = ce.score_loaded(m, tok, dv, c[sel][["s1_id", "cand_id"]].reset_index(drop=True),
                                                     left, right, batch=256)
        c.to_parquet(CE_OUT / "test_ce" / (f.name + ".tmp"), index=False)
        (CE_OUT / "test_ce" / (f.name + ".tmp")).replace(CE_OUT / "test_ce" / f.name)
        print(f"{f.name}: re-scored {int(sel.sum()):,} unseen-country rows", flush=True)


if __name__ == "__main__":
    {"train": cmd_train, "check": cmd_check, "score": cmd_score}[sys.argv[1]]()
