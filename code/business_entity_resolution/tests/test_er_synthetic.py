"""Synthetic ER data matches the organiser format and the loaders/validator/metric accept it."""
from src.er_submission import load_ground_truth, load_sources, validate_outputs, write_outputs
from src.er_synthetic import generate
from src.metrics import er_fbeta_macro


def test_synthetic_roundtrip(tmp_path):
    root = generate(tmp_path / "synth", n=60, seed=1)
    tr1, tr2, tr3 = load_sources(root / "dataset/train")
    te1, te2, te3 = load_sources(root / "dataset/test")
    assert set(tr1["country"]) <= {"US", "India"} and "France" in set(te1["country"])
    assert list(tr1.columns) == ["entity_id", "business_name", "business_address", "country"]
    gt = load_ground_truth(root / "dataset/train/train_ground_truth.tsv")
    assert set(gt) == set(tr1["entity_id"]) and any(not v for v in gt.values())   # singletons exist
    all_ids = set(tr2["entity_id"]) | set(tr3["entity_id"])
    assert all(v <= all_ids for v in gt.values())

    # oracle predictions score 1.0; all-empty scores exactly the singleton rate
    test_gt = load_ground_truth(root / "dataset/test/test_ground_truth.tsv")
    assert er_fbeta_macro(test_gt, test_gt) == 1.0
    singleton_rate = sum(not v for v in test_gt.values()) / len(test_gt)
    assert er_fbeta_macro(test_gt, {}) == singleton_rate

    rep = write_outputs(test_gt, test_gt, te1["entity_id"], tmp_path / "out", test_dir=root / "dataset/test")
    assert rep.ok and validate_outputs(tmp_path / "out/matching_results.tsv", test_dir=root / "dataset/test").ok


def test_synthetic_is_deterministic(tmp_path):
    a = generate(tmp_path / "a", n=30, seed=7)
    b = generate(tmp_path / "b", n=30, seed=7)
    for f in ("train/train_source2.tsv", "test/test_ground_truth.tsv"):
        assert (a / "dataset" / f).read_text(encoding="utf-8") == (b / "dataset" / f).read_text(encoding="utf-8")
