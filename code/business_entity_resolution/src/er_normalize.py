"""Text normalisation for business names and addresses (contract: docs/WORKERS.md).

Pure str -> str / list functions, never branching on specific country values (France is unseen in train).
Every non-Latin script is transliterated to ASCII with anyascii (rule-based code, not external data).

    from src.er_normalize import normalize_name, core_name, normalize_address, extract_numbers, extract_postcode
    normalize_frame(df)   # adds name_norm, name_core, addr_norm, addr_nums, postcode (parallel over 32 cores)
"""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from anyascii import anyascii

# Legal-form tokens across the scripts we see, after transliteration. Generic word lists, not a data lookup.
LEGAL_TOKENS = frozenset("""
pvt private pvtltd ltd limited llp llc inc incorporated corp corporation co company cos
plc lp pllc pc sa sas sasu sarl eurl sci snc gmbh ag bv nv oy ab spa srl
praivet praiveta praaivet limited limiteda limitid limitad
""".split())
# Multi-token legal phrases collapse before token filtering (e.g. "l l p" after punctuation stripping).
_LEGAL_PHRASES = [(re.compile(r"\bl l p\b"), "llp"), (re.compile(r"\bl l c\b"), "llc"),
                  (re.compile(r"\bp\s?ltd\b"), "pvt ltd"), (re.compile(r"\bf k a\b"), "fka"),
                  (re.compile(r"\bd b a\b"), "dba")]
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_NULL_TOKENS = re.compile(r"\b(null|none|nan|n a)\b")
_DIGITS = re.compile(r"\d+")
_POSTCODE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")
_SPACES = re.compile(r"\s+")


def to_ascii(s: str) -> str:
    """Transliterate any script to ASCII and lowercase ('राम' -> 'ram', 'Léarning' -> 'learning')."""
    return anyascii(s or "").lower()


def _clean(s: str) -> str:
    """ASCII-transliterate, replace '&' by 'and', drop punctuation and literal nulls, squeeze spaces."""
    s = to_ascii(s).replace("&", " and ")
    s = _NON_ALNUM.sub(" ", s)
    s = _NULL_TOKENS.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def normalize_name(s: str) -> str:
    """Lowercase ASCII name with punctuation removed and legal phrases unified."""
    s = _clean(s)
    for pat, rep in _LEGAL_PHRASES:
        s = pat.sub(rep, s)
    return s


def core_name(s: str) -> str:
    """Name without legal-form tokens, 'and', or former-name markers; falls back to the full name if nothing is left."""
    toks = normalize_name(s).split()
    if "fka" in toks:  # "Veonexx F/K/A Corey Bright Inc": keep both names, drop the marker
        toks = [t for t in toks if t != "fka"]
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in ("and", "the", "dba")]
    return " ".join(core) if core else " ".join(toks)


def normalize_address(s: str) -> str:
    """Lowercase ASCII address, punctuation removed, literal 'null' dropped."""
    return _clean(s)


def extract_numbers(s: str) -> list[str]:
    """All digit runs in order (house/plot numbers, postcodes), leading zeros stripped ('004669' -> '4669')."""
    return [d.lstrip("0") or "0" for d in _DIGITS.findall(to_ascii(s))]


def extract_postcode(s: str) -> str:
    """Last standalone 5- or 6-digit run (US ZIP / India PIN / FR code postal), '' if none."""
    m = _POSTCODE.findall(to_ascii(s))
    return m[-1] if m else ""


def _norm_chunk(args: tuple[list[str], list[str]]) -> dict[str, list]:
    """Normalise one chunk of names and addresses (worker function)."""
    names, addrs = args
    return {
        "name_norm": [normalize_name(x) for x in names],
        "name_core": [core_name(x) for x in names],
        "addr_norm": [normalize_address(x) for x in addrs],
        "addr_nums": [" ".join(extract_numbers(x)) for x in addrs],
        "postcode": [extract_postcode(x) for x in addrs],
    }


def normalize_frame(df: pd.DataFrame, workers: int = 30, chunk: int = 200_000) -> pd.DataFrame:
    """Return df with normalised columns added (row order preserved). Parallel over processes."""
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    if len(parts) == 1 or workers <= 1:
        results = [_norm_chunk(p) for p in parts]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_norm_chunk, parts))
    out = df.copy()
    for col in results[0]:
        out[col] = [v for r in results for v in r[col]]
    return out


def main() -> None:
    """Normalise every cached source file once: data/cache/<split>_s<k>_norm.parquet (same row order)."""
    import argparse
    import time

    from src.er_data import cache_dir

    ap = argparse.ArgumentParser(prog="python -m src.er_normalize")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    for split in a.splits:
        for k in (1, 2, 3):
            t = time.time()
            df = pd.read_parquet(cache_dir() / f"{split}_s{k}.parquet")
            out = normalize_frame(df, workers=a.workers)
            out.drop(columns=["business_name", "business_address"]).to_parquet(
                cache_dir() / f"{split}_s{k}_norm.parquet", index=False)
            print(f"{split} S{k}: {len(df):,} rows in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
