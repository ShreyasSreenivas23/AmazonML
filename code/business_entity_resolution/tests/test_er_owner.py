"""Listwise owner model: grouping, labels, leakage filter, sequence layout, and a tiny-encoder forward/backward."""
import numpy as np
import pandas as pd
import pytest

from src.er_owner import K_MAX, build_groups, collate, exclude_fold0, label_groups, make_example, owner_model


def pairs():
    return pd.DataFrame({
        "s1_id": ["S1-a", "S1-b", "S1-c", "S1-a", "S1-b", "S1-d"],
        "cand_id": ["S2-x", "S2-x", "S2-x", "S3-y", "S3-y", "S2-z"],
        "prob": [0.9, 0.3, 0.004, 0.8, 0.02, 0.7],
    })


def test_build_groups_contest_and_order():
    g = build_groups(pairs(), k_max=6, min_prob=0.01, contest_min=0.05).set_index("cand_id")
    assert list(g.index) == ["S2-x"]                        # y: 2nd prob 0.02 < 0.05; z: single S1
    assert g.loc["S2-x", "s1_ids"] == ["S1-a", "S1-b"]       # c dropped by min_prob, sorted by prob
    assert build_groups(pairs(), k_max=1, min_prob=0.0, contest_min=0.0).empty   # k_max 1 -> never contested


def test_labels_owner_slot_or_none():
    g = build_groups(pairs(), min_prob=0.0, contest_min=0.0)
    owner = pd.Series({"S2-x": "S1-b", "S3-y": "S1-q"})
    lab = dict(zip(g.cand_id, label_groups(g, owner)))
    assert lab["S2-x"] == 1 and lab["S3-y"] == K_MAX          # owner outside the list -> none slot


def test_exclude_fold0_uses_slot_lists_and_owner():
    g = pd.DataFrame({"cand_id": ["r1", "r2", "r3"],
                      "s1_ids": [["S1-a", "S1-b"], ["S1-b", "S1-c"], ["S1-c", "S1-d"]]})
    fold = pd.Series({"S1-a": 0, "S1-b": 1, "S1-c": 2, "S1-d": 3, "S1-e": 0})
    owner = pd.Series({"r3": "S1-e"})                         # r3's owner is a fold-0 S1 outside its list
    assert list(exclude_fold0(g, owner, fold).cand_id) == ["r2"]


def test_make_example_layout():
    ids, seg = make_example([7, 8], [[5], [6, 6]], bos=0, eos=2)
    assert ids == [0, 7, 8, 2, 2, 5, 2, 2, 6, 6, 2]
    assert seg == [0, 1, 1, 0, 0, 2, 0, 0, 3, 3, 0]


def test_tiny_model_forward_backward():
    torch = pytest.importorskip("torch")
    from transformers import XLMRobertaConfig, XLMRobertaModel

    enc = XLMRobertaModel(XLMRobertaConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                           intermediate_size=32, max_position_embeddings=80, pad_token_id=1))
    m = owner_model(enc, 16)
    ex = [make_example([7, 8], [[5], [6, 6]], 0, 2), make_example([9], [[5], [4], [3]], 0, 2)]
    ids, att, seg = collate(ex, pad=1)
    logits = m(ids, att, seg, torch.tensor([2, 3]))
    assert logits.shape == (2, K_MAX + 1)
    assert (logits[0, 2:K_MAX] < -1e3).all() and (logits[1, 3:K_MAX] < -1e3).all()   # empty slots masked
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, K_MAX]))
    loss.backward()
    assert np.isfinite(loss.item()) and m.slot[0].weight.grad is not None
