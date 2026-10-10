"""Pair features on synthetic ER data: shapes, ordering, and that true pairs look more similar than random ones."""
import numpy as np
import pandas as pd

from src.er_features import FEATURE_GROUPS, build_features
from src.er_normalize import normalize_frame
from src.er_submission import load_ground_truth, load_sources
from src.er_synthetic import generate


def _setup(tmp_path):
    root = generate(tmp_path / "d", n=80, seed=5)
    s1, s2, s3 = load_sources(root / "dataset/train")
    gt = load_ground_truth(root / "dataset/train/train_ground_truth.tsv")
    left = normalize_frame(s1, workers=1).set_index("entity_id")
    right = normalize_frame(pd.concat([s2, s3], ignore_index=True), workers=1).set_index("entity_id")
    pos = pd.DataFrame([(k, c) for k, v in gt.items() for c in v], columns=["s1_id", "cand_id"])
    rng = np.random.default_rng(0)
    neg = pd.DataFrame({"s1_id": pos.s1_id.to_numpy(), "cand_id": rng.permutation(right.index.to_numpy())[:len(pos)]})
    pairs = pd.concat([pos.assign(y=1), neg.assign(y=0)], ignore_index=True)
    return pairs, left, right


def test_features_shape_order_and_signal(tmp_path):
    pairs, left, right = _setup(tmp_path)
    d = 8
    rng = np.random.default_rng(1)
    emb = {v: (rng.normal(size=(len(left), d)).astype(np.float16), rng.normal(size=(len(right), d)).astype(np.float16))
           for v in ("name", "addr", "both")}
    X = build_features(pairs.drop(columns="y"), left, right, emb)
    assert len(X) == len(pairs) and list(X.s1_id) == list(pairs.s1_id) and list(X.cand_id) == list(pairs.cand_id)
    feats = X.drop(columns=["s1_id", "cand_id"])
    assert (feats.dtypes == np.float32).all()
    assert {"cos_both", "name_tset", "addr_tset", "num_jaccard", "ctx_rank_in_s1", "ctx_rank_in_cand"} <= set(feats)
    pos, neg = X[pairs.y.to_numpy() == 1], X[pairs.y.to_numpy() == 0]
    assert pos.name_tset.mean() > neg.name_tset.mean() + 0.2
    assert pos.addr_tset.mean() > neg.addr_tset.mean() + 0.2
    assert not any(c.lower().startswith("country") for c in feats)   # no country identity


def test_groups_registry_and_subset(tmp_path):
    pairs, left, right = _setup(tmp_path)
    assert {"embed", "name", "address", "context"} <= set(FEATURE_GROUPS)
    X = build_features(pairs.drop(columns="y"), left, right, groups=["name"])
    assert "name_tset" in X and "addr_tset" not in X
