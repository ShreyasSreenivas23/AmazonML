"""Parquet cache builder on synthetic data in the organiser layout."""
from src.er_synthetic import generate


def test_build_cache_and_load(tmp_path, monkeypatch):
    root = generate(tmp_path / "d", n=40, seed=3)
    monkeypatch.setenv("DATA_DIR", str(root))
    from src.er_data import build_cache, load

    build_cache("train")
    build_cache("test")
    s1, s2, s3, gt = load("train")
    assert list(s1.columns) == ["entity_id", "business_name", "business_address", "country"]
    assert len(s1) == 40 and gt is not None and set(gt.columns) == {"s1_id", "cand_id"}
    assert gt.cand_id.str[:2].isin(["S2", "S3"]).all() and (gt.cand_id != "").all()
    t1, _, _, tgt = load("test", with_gt=False)
    assert tgt is None and "France" in set(t1.country)
