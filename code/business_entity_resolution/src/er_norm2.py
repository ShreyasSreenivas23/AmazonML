"""Normalisation v2 (E024): extra, cleaner name/address strings for stage-2 features, built on top of er_normalize.

Why (26 Sep, French test records inspected): street types are never expanded, so "2 R PAUL VERLANE" and
"2 RUE PAUL VERLAINE" (or US "St"/"Street", "Ave"/"Avenue") look different to every address similarity; dotted legal
forms such as "S.A.S" survive as "s a s" in the core name. Both are fixed with small hand-written, language-generic
dictionaries (explicitly allowed, docs/COMPETITION.md). Nothing branches on the country value.

    python -m src.er_norm2          # data/cache/<split>_s<k>_norm2.parquet: entity_id, name_core2, addr_norm2

The v1 columns are left untouched, so every existing feature and cached table stays valid; v2 only adds features.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.er_normalize import LEGAL_TOKENS

# Street-type and address abbreviations seen in the three countries' records (generic word list, not a data lookup).
ADDR_ABBR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "avn": "avenue", "bd": "boulevard", "blvd": "boulevard",
    "bld": "boulevard", "imp": "impasse", "crs": "cours", "pl": "place", "ch": "chemin", "che": "chemin",
    "chem": "chemin", "rte": "route", "all": "allee", "sq": "square", "fbg": "faubourg", "qu": "quai",
    "st": "street", "str": "street", "rd": "road", "ln": "lane", "dr": "drive", "ct": "court", "cir": "circle",
    "pkwy": "parkway", "hwy": "highway", "ter": "terrace", "trl": "trail", "pl.": "place",
    "apt": "apartment", "ste": "suite", "fl": "floor", "bldg": "building", "flr": "floor",
    "nr": "near", "opp": "opposite", "mkt": "market", "rly": "railway", "stn": "station",
}
# Extra legal-form / company-word tokens, on top of er_normalize.LEGAL_TOKENS.
LEGAL2 = LEGAL_TOKENS | frozenset("ei eirl selarl sca scop scs sasu cie ets etablissements societe ste".split())
DROP2 = frozenset(("and", "the", "dba", "et", "fka"))


def collapse_initials(toks: list[str]) -> list[str]:
    """Join runs of >= 2 single-letter tokens: ['s','a','s'] -> ['sas'], ['s','a','r','l'] -> ['sarl']."""
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


def core_name2(name_norm: str) -> str:
    """v1 normalised name -> core name with initials collapsed and a wider legal-word list removed."""
    toks = collapse_initials(str(name_norm).split())
    core = [t for t in toks if t not in LEGAL2 and t not in DROP2]
    return " ".join(core) if core else " ".join(toks)


def addr_norm2(addr_norm: str) -> str:
    """v1 normalised address -> street-type abbreviations expanded (token-wise, so 'r' only as a whole word)."""
    return " ".join(ADDR_ABBR.get(t, t) for t in str(addr_norm).split())


def _chunk(args: tuple[list[str], list[str]]) -> tuple[list[str], list[str]]:
    """Normalise one chunk of names and addresses (worker function)."""
    names, addrs = args
    return [core_name2(x) for x in names], [addr_norm2(x) for x in addrs]


def norm2_frame(n: pd.DataFrame, workers: int = 30, chunk: int = 250_000) -> pd.DataFrame:
    """From a v1 *_norm frame: entity_id, name_core2, addr_norm2 (row order preserved)."""
    names, addrs = n["name_norm"].tolist(), n["addr_norm"].tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(n), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_chunk, parts))
    return pd.DataFrame({"entity_id": n["entity_id"].to_numpy(),
                         "name_core2": [v for r in res for v in r[0]],
                         "addr_norm2": [v for r in res for v in r[1]]})


def main() -> None:
    """Write v2 normalisation caches for every source of train and test."""
    import time

    from src.er_data import cache_dir

    for split in ("train", "test"):
        for k in (1, 2, 3):
            t = time.time()
            n = pd.read_parquet(cache_dir() / f"{split}_s{k}_norm.parquet", columns=["entity_id", "name_norm", "addr_norm"])
            norm2_frame(n).to_parquet(cache_dir() / f"{split}_s{k}_norm2.parquet", index=False)
            print(f"{split} S{k}: {len(n):,} rows [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
