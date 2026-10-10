"""Normalisation behaviour on noise patterns seen in EDA (docs/DECISIONS.md, P0)."""
import pandas as pd

from src.er_normalize import (
    core_name,
    extract_numbers,
    extract_postcode,
    normalize_address,
    normalize_frame,
    normalize_name,
)


def test_names_cross_script_and_punctuation():
    assert normalize_name("SU-SECURITY L.L.P.") == "su security llp"
    assert normalize_name("Powell, Dunn & Propst") == "powell dunn and propst"
    assert normalize_name("-- Holloway Peak Inc Seafood") == "holloway peak inc seafood"
    assert normalize_name("LLC Moncada Léarning Center") == "llc moncada learning center"
    assert normalize_name("एसएस फूड").isascii()


def test_core_name_drops_legal_forms_anywhere():
    assert core_name("Ss Food Private Limited") == "ss food"
    assert core_name("SS FOOD PVT. LTD.") == "ss food"
    assert core_name("LLC Moncada Learning Center") == "moncada learning center"
    assert core_name("Chez Paul SARL") == "chez paul"
    assert core_name("Veonexx F/K/A Corey Bright Inc") == "veonexx corey bright"
    assert core_name("Limited") == "limited"  # nothing left -> keep the full name


def test_address_numbers_and_postcodes():
    assert normalize_address("#298 E-18/59 G/f, null, Delhi") == "298 e 18 59 g f delhi"
    assert extract_numbers("004669 3502, Wills Point") == ["4669", "3502"]
    assert extract_postcode("Tigard, OR 97223") == "97223"
    assert extract_postcode("Pune 411001") == "411001"
    assert extract_postcode("12 Rue de Rivoli, 75001 Paris") == "75001"
    assert extract_postcode("1708 Landmark Drive") == ""


def test_normalize_frame_matches_scalar_and_keeps_order():
    df = pd.DataFrame({"entity_id": ["a", "b", "c"],
                       "business_name": ["Acme Pvt Ltd", "राम मार्केटिंग", "Chez Paul SARL"],
                       "business_address": ["12 Main St 94107", "", "3 Rue X, 75001 Paris"],
                       "country": ["US", "India", "France"]})
    for workers in (1, 2):
        out = normalize_frame(df, workers=workers, chunk=1)
        assert list(out.entity_id) == ["a", "b", "c"]
        assert list(out.name_core) == [core_name(x) for x in df.business_name]
        assert list(out.postcode) == ["94107", "", "75001"]
