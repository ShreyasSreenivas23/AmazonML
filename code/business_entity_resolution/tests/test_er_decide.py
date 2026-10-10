"""Decision layer: thresholds, assignment constraint, fallback, expected-F prefix."""
import numpy as np
import pandas as pd

from src.er_decide import decide, decide_expected_f, expected_fbeta_prefix, tune_threshold


def _p():
    return pd.DataFrame({
        "s1_id": ["A", "A", "A", "B", "B", "C"],
        "cand_id": ["S2-1", "S3-1", "S2-9", "S2-9", "S3-5", "S2-7"],
        "prob": [0.95, 0.80, 0.40, 0.90, 0.20, 0.30],
    })


def test_threshold_and_assignment():
    p = _p()
    assert decide(p, 0.5, assign=False) == {"A": {"S2-1", "S3-1"}, "B": {"S2-9"}}
    # S2-9 appears under A (0.40) and B (0.90): assignment keeps it only for B
    assert decide(p, 0.3, assign=True) == {"A": {"S2-1", "S3-1"}, "B": {"S2-9"}, "C": {"S2-7"}}
    assert "S2-9" in decide(p, 0.3, assign=False)["A"]


def test_top1_fallback_only_for_empty_entities():
    out = decide(_p(), threshold=0.85, assign=True, top1_fallback=0.25)
    assert out["A"] == {"S2-1"} and out["B"] == {"S2-9"} and out["C"] == {"S2-7"}   # C via fallback
    assert "C" not in decide(_p(), threshold=0.85, assign=True, top1_fallback=0.35)


def test_expected_f_prefix_behaviour():
    assert expected_fbeta_prefix([]) == 0
    assert expected_fbeta_prefix([0.02, 0.01]) == 0          # likely a singleton -> predict empty
    assert expected_fbeta_prefix([0.99, 0.98, 0.97]) == 3
    assert expected_fbeta_prefix([0.97, 0.15, 0.1]) == 1      # precision-heavy: stop after the sure one
    out = decide_expected_f(_p())
    assert out["A"] >= {"S2-1"} and "S2-9" not in out.get("A", set())


def test_tune_threshold_counts_all_entities():
    y = {"A": {"S2-1", "S3-1"}, "B": {"S2-9"}, "C": set(), "D": set()}   # D has no candidates at all
    t, f, tab = tune_threshold(_p(), y, grid=[0.25, 0.5, 0.85])
    assert t == 0.5 and np.isclose(f, 1.0) and len(tab) == 3
