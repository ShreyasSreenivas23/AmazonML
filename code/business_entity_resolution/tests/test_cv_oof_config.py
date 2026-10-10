import numpy as np
import pandas as pd
import pytest

from src.config import load_config, save_config
from src.cv import iter_folds, make_folds
from src.oof import load_oof, save_oof, stack_oofs


@pytest.fixture
def df():
    rng = np.random.RandomState(0)
    n = 500
    return pd.DataFrame({
        "id": np.arange(n) + 1000,
        "y": rng.lognormal(size=n),
        "cls": rng.randint(0, 3, n),
        "grp": rng.randint(0, 60, n),
        "t": rng.rand(n),
    })


@pytest.mark.parametrize("method,kw", [
    ("kfold", {}),
    ("stratified", {"target": "cls"}),
    ("stratified_reg", {"target": "y"}),
    ("group", {"group": "grp"}),
    ("stratified_group", {"target": "cls", "group": "grp"}),
    ("time", {"time_col": "t"}),
])
def test_folds_cover_all_rows(df, method, kw):
    f = make_folds(df, n_splits=5, method=method, seed=1, **kw)
    assert len(f) == len(df) and set(f["fold"]) == set(range(5))
    assert f["id"].tolist() == df["id"].tolist()
    if "group" in kw:  # no group spans two folds
        assert (df.assign(fold=f["fold"]).groupby("grp")["fold"].nunique() == 1).all()


def test_folds_reproducible(df):
    a = make_folds(df, method="stratified_reg", target="y", seed=7)
    b = make_folds(df, method="stratified_reg", target="y", seed=7)
    assert a.equals(b)


def test_iter_folds_disjoint(df):
    f = make_folds(df, n_splits=4)
    for _, tr, va in iter_folds(f):
        assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == len(df)


def test_oof_roundtrip_and_stack(df, tmp_path):
    f = make_folds(df, n_splits=5)
    test_ids = np.arange(100)
    for e in ("E001", "E002"):
        save_oof(e, df["id"], np.random.rand(len(df)), f["fold"], test_ids, np.random.rand(100),
                 metric="smape", cv=12.3, root=tmp_path)
    oof, test, meta = load_oof("E001", tmp_path)
    assert len(oof) == len(df) and meta["cv"] == 12.3 and len(test) == 100
    X, Xt = stack_oofs(["E001", "E002"], tmp_path)
    assert list(X.columns) == ["id", "fold", "E001__pred", "E002__pred"]
    assert Xt.shape == (100, 3)


def test_oof_rejects_mismatched_folds(df, tmp_path):
    save_oof("A", df["id"], np.zeros(len(df)), make_folds(df, seed=1)["fold"], root=tmp_path)
    save_oof("B", df["id"], np.zeros(len(df)), make_folds(df, seed=2)["fold"], root=tmp_path)
    with pytest.raises(ValueError, match="folds differ"):
        stack_oofs(["A", "B"], tmp_path)


def test_oof_rejects_nan(df, tmp_path):
    p = np.zeros(len(df))
    p[3] = np.nan
    with pytest.raises(ValueError):
        save_oof("X", df["id"], p, np.zeros(len(df), int), root=tmp_path)


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\ntrain: {lr: 0.1, epochs: 3}\n")
    (tmp_path / "child.yaml").write_text("defaults: base.yaml\ntrain: {lr: 0.01}\nname: x\n")
    cfg = load_config(tmp_path / "child.yaml", ["train.epochs=5", "new.key=[1, 2]"])
    assert cfg.seed == 1 and cfg.train.lr == 0.01 and cfg.train.epochs == 5 and cfg.new.key == [1, 2]
    save_config(cfg, tmp_path / "out.yaml")
    assert load_config(tmp_path / "out.yaml").train.epochs == 5


def test_repo_configs_load():
    cfg = load_config("configs/example.yaml")
    assert cfg.exp_id and cfg.cv.n_splits >= 2
