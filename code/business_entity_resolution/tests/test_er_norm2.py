"""Normalisation v2: street-type expansion and initial collapsing (country-agnostic)."""
from src.er_norm2 import addr_norm2, collapse_initials, core_name2


def test_addresses_expand_street_types_as_whole_words():
    assert addr_norm2("2 r paul verlane saint nazaire") == "2 rue paul verlane saint nazaire"
    assert addr_norm2("12 main st apt 4") == "12 main street apartment 4"
    assert addr_norm2("22 imp du piqcey") == "22 impasse du piqcey"
    assert addr_norm2("rue r") == "rue rue" and addr_norm2("rosebud rd") == "rosebud road"   # whole tokens only


def test_core_name_collapses_initials_and_drops_legal_words():
    assert collapse_initials(["belsunce", "ecole", "s", "a", "s"]) == ["belsunce", "ecole", "sas"]
    assert core_name2("belsunce ecole s a s") == "belsunce ecole"
    assert core_name2("coast maternelle france ei") == "coast maternelle france"
    assert core_name2("x y testing") == "xy testing"            # generic rule, not a French special case
    assert core_name2("sarl") == "sarl"                          # never empties a name


def test_norm2_pair_features_see_expanded_streets():
    import pandas as pd

    from src.er_fullpass import norm2_pair_features
    L2 = pd.DataFrame({"name_core2": ["belsunce ecole"], "addr_norm2": ["3 rue racine dunkerque"]}, index=["S1-1"])
    R2 = pd.DataFrame({"name_core2": ["belsunce ecole", "other"], "addr_norm2": ["3 rue racine", "9 main street"]},
                      index=["S2-a", "S2-b"])
    f = norm2_pair_features(["S1-1", "S1-1"], ["S2-a", "S2-b"], L2, R2)
    assert set(f) == {"name2_ratio", "name2_tset", "name2_tsort", "addr2_tset", "addr2_tsort"}
    assert f["name2_ratio"][0] == 1.0 and f["addr2_tset"][0] == 1.0 and f["addr2_tset"][1] < 0.5
