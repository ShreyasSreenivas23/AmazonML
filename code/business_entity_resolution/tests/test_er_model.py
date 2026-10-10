"""LightGBM OOF training on a separable toy problem."""
import numpy as np
import pandas as pd

from src.er_model import importance, predict, train_oof


def test_train_oof_and_predict():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"s1_id": [f"S1-{i // 4}" for i in range(n)], "cand_id": [f"S2-{i}" for i in range(n)],
                      "good": rng.random(n).astype(np.float32), "noise": rng.random(n).astype(np.float32)})
    y = (X.good > 0.6).to_numpy().astype(int)
    groups = (np.arange(n) // 4) % 3
    oof, models = train_oof(X, y, groups, {"num_leaves": 7, "min_data_in_leaf": 20, "num_threads": 2},
                            num_boost_round=100, early_stopping=10)
    assert len(models) == 3 and not np.isnan(oof).any()
    assert ((oof > 0.5) == y).mean() > 0.97
    assert importance(models).index[0] == "good"
    assert predict(models, X).shape == (n,)
