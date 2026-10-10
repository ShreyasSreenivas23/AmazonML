import math

import numpy as np
import pytest

from src.metrics import get_metric, list_metrics


def m(name):
    return get_metric(name)[0]


def test_registry_directions():
    assert get_metric("smape")[1] is False
    assert get_metric("f1_macro")[1] is True
    assert "extraction_f1" in list_metrics()
    with pytest.raises(KeyError):
        get_metric("nope")


def test_smape_hand_computed():
    # |110-100| / ((100+110)/2) = 10/105 ; |50-100|/((100+50)/2)=50/75
    exp = 100 * (10 / 105 + 50 / 75) / 2
    assert m("smape")([100, 100], [110, 50]) == pytest.approx(exp)


def test_smape_edges():
    assert m("smape")([0, 0], [0, 0]) == 0.0          # 0/0 -> 0
    assert m("smape")([0], [5]) == pytest.approx(200)  # max error
    assert m("smape")([5], [5]) == 0.0
    assert m("smape")([1, 2, 3], [1, 2, 3]) == 0.0
    # symmetric in the "symmetric" sense
    assert m("smape")([100], [120]) == pytest.approx(m("smape")([120], [100]))


def test_smape_rejects_nan_and_shape():
    with pytest.raises(ValueError):
        m("smape")([1, 2], [1, np.nan])
    with pytest.raises(ValueError):
        m("smape")([1, 2], [1])


def test_regression_basics():
    y, p = [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]
    assert m("rmse")(y, p) == pytest.approx(math.sqrt(4 / 3))
    assert m("mae")(y, p) == pytest.approx(2 / 3)
    assert m("rmsle")([0, math.e - 1], [0, 0]) == pytest.approx(math.sqrt(0.5))
    assert m("r2")(y, y) == pytest.approx(1.0)
    assert m("mape")([0, 100], [5, 110]) == pytest.approx(10.0)  # zero target excluded


def test_classification():
    assert m("accuracy")([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert m("f1_macro")([0, 1], [0, 1]) == 1.0
    assert m("auc")([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_extraction_f1_cases():
    gt = ["10 gram", "", "5 cm", "2 kg", ""]
    out = ["10 gram", "", "6 cm", "", "1 kg"]
    # TP=1, FP=2 (wrong value + pred on empty gt), FN=1, TN=1
    p, r = 1 / 3, 1 / 2
    assert m("extraction_f1")(gt, out) == pytest.approx(2 * p * r / (p + r))
    assert m("extraction_f1")(["a"], ["a"]) == 1.0
    assert m("extraction_f1")([""], [""]) == 0.0


def test_ranking():
    assert m("map_at_k")([[1, 2]], [[1, 2, 3]], k=3) == 1.0
    assert m("map_at_k")([[3]], [[1, 2, 3]], k=3) == pytest.approx(1 / 3)
    assert m("ndcg")([[3, 2, 1]], [[0.9, 0.5, 0.1]]) == pytest.approx(1.0)


# ---------------------------------------------------------------- 2026 entity resolution F0.5
from src.metrics import entity_fbeta, er_fbeta_macro, parse_id_list  # noqa: E402


def test_er_statement_example():
    # verbatim example from the problem statement -> 0.714
    f = entity_fbeta({"S2-00047", "S3-00812"}, {"S2-00047", "S2-00193", "S3-00812"})
    assert f == pytest.approx(0.7142857, abs=1e-6)
    assert round(f, 3) == 0.714


def test_er_singleton_and_edge_cases():
    assert entity_fbeta(set(), set()) == 1.0            # correct singleton
    assert entity_fbeta(set(), {"S2-1"}) == 0.0         # false merge on a singleton
    assert entity_fbeta({"S2-1"}, set()) == 0.0         # missed everything
    assert entity_fbeta({"S2-1"}, {"S3-9"}) == 0.0      # wrong match
    assert entity_fbeta({"S2-1", "S3-2"}, {"S2-1", "S3-2"}) == 1.0
    # precision-heavy: 1 of 2 true found with no FP beats 2 of 2 found with 2 FP
    assert entity_fbeta({"a", "b"}, {"a"}) > entity_fbeta({"a", "b"}, {"a", "b", "c", "d"})
    assert entity_fbeta({"a", "b"}, {"a"}) == pytest.approx(1.25 * 0.5 / (0.25 + 0.5))


def test_er_macro_average_and_missing_keys():
    y_true = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-3", "S3-3"}}
    y_pred = {"S1-1": {"S2-1"}, "S1-3": {"S2-3"}}       # S1-2 missing -> empty -> correct singleton
    exp = (1.0 + 1.0 + 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5)) / 3
    assert er_fbeta_macro(y_true, y_pred) == pytest.approx(exp)
    assert get_metric("er_f05")[1] is True


def test_parse_id_list():
    assert parse_id_list("S2-1,S3-4") == {"S2-1", "S3-4"}
    assert parse_id_list(" S2-1 , S2-1 ,") == {"S2-1"}
    assert parse_id_list("") == set() and parse_id_list(None) == set() and parse_id_list(float("nan")) == set()
