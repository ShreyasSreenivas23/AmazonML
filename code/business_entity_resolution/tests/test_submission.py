import numpy as np
import pandas as pd
import pytest

from src.submission import main, validate_submission, write_submission


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"sample_id": [f"{i:03d}" for i in range(10)], "price": 0.0}).to_csv(p, index=False)
    return p


def _write(tmp_path, df, name="sub.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_valid_roundtrip(tmp_path, sample):
    ids = [f"{i:03d}" for i in range(10)][::-1]  # shuffled order -> writer reorders to sample
    rep = write_submission(ids, np.arange(10) + 1.0, tmp_path / "s.csv", "sample_id", "price", sample=sample)
    assert rep.ok
    assert pd.read_csv(tmp_path / "s.csv", dtype=str)["sample_id"].tolist() == [f"{i:03d}" for i in range(10)]


def test_catches_nan_missing_dupes_and_index(tmp_path, sample):
    ids = [f"{i:03d}" for i in range(10)]
    bad = pd.DataFrame({"sample_id": ids, "price": [1.0] * 9 + [np.nan]})
    assert not validate_submission(_write(tmp_path, bad), sample).ok
    short = pd.DataFrame({"sample_id": ids[:9], "price": 1.0})
    rep = validate_submission(_write(tmp_path, short), sample)
    assert any("row count" in e for e in rep.errors)
    dup = pd.DataFrame({"sample_id": ids[:9] + ids[:1], "price": 1.0})
    assert any("duplicate" in e for e in validate_submission(_write(tmp_path, dup), sample).errors)
    p = tmp_path / "idx.csv"
    pd.DataFrame({"sample_id": ids, "price": 1.0}).to_csv(p)  # index written
    assert not validate_submission(p, sample).ok


def test_leading_zero_ids_not_coerced(tmp_path, sample):
    # ids written as ints lose leading zeros -> must fail, not silently pass
    df = pd.DataFrame({"sample_id": list(range(10)), "price": 1.0})
    assert not validate_submission(_write(tmp_path, df), sample).ok


def test_order_and_negative(tmp_path, sample):
    ids = [f"{i:03d}" for i in range(10)]
    rev = pd.DataFrame({"sample_id": ids[::-1], "price": 1.0})
    p = _write(tmp_path, rev)
    assert not validate_submission(p, sample).ok
    assert validate_submission(p, sample, allow_reorder=True).ok
    neg = pd.DataFrame({"sample_id": ids, "price": -1.0})
    assert not validate_submission(_write(tmp_path, neg), sample, nonneg=True).ok


def test_cli_exit_codes(tmp_path, sample):
    good = pd.DataFrame({"sample_id": [f"{i:03d}" for i in range(10)], "price": 2.0})
    assert main(["validate", str(_write(tmp_path, good)), "--sample", str(sample)]) == 0
    assert main(["validate", str(tmp_path / "missing.csv"), "--sample", str(sample)]) == 1
