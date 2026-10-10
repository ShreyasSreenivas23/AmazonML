"""Pseudo-label selection for cross-encoder self-training (synthetic, no GPU)."""
import pandas as pd

from src.er_selftrain import pseudo_pairs


def test_pseudo_pairs_country_thresholds_and_hard_negatives(tmp_path):
    (tmp_path / "chunks").mkdir()
    pd.DataFrame({"s1_id": ["A", "A", "A", "B", "B", "C"], "cand_id": ["a1", "a2", "a3", "b1", "b2", "c1"],
                  "prob": [0.9, 0.8, 0.01, 0.9, 0.7, 0.9]}).to_parquet(tmp_path / "chunks" / "000.parquet")
    probs = pd.DataFrame({"s1_id": ["A", "A", "A", "B", "B", "C"], "cand_id": ["a1", "a2", "a3", "b1", "b2", "c1"],
                          "prob": [0.99, 0.02, 0.01, 0.5, 0.98, 0.99]})
    country = pd.Series({"A": "France", "B": "France", "C": "US"})
    out = pseudo_pairs(probs, str(tmp_path / "chunks"), country, "France", 0.97, 0.03, n_pos=10, n_neg=1)
    assert "C" not in set(out.s1_id)                                   # other countries excluded
    assert set(zip(out.s1_id, out.cand_id, out.y)) == {("A", "a1", True), ("B", "b2", True), ("A", "a2", False)}
    # uncertain b1 (0.5) never used; the single negative kept is the hard one (stage-1 prob 0.8 beats 0.01)


def test_pseudo_pairs_accepts_several_countries(tmp_path):
    (tmp_path / "chunks").mkdir()
    pd.DataFrame({"s1_id": ["A", "C"], "cand_id": ["a1", "c1"], "prob": [0.9, 0.9]}).to_parquet(tmp_path / "chunks" / "000.parquet")
    probs = pd.DataFrame({"s1_id": ["A", "C"], "cand_id": ["a1", "c1"], "prob": [0.99, 0.99]})
    country = pd.Series({"A": "India", "C": "US"})
    out = pseudo_pairs(probs, str(tmp_path / "chunks"), country, ["US", "India"], 0.97, 0.03, n_pos=10, n_neg=10)
    assert set(out.s1_id) == {"A", "C"}
