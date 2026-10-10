"""Cross-encoder: sampling, train/score end to end, resume and CLI on a tiny random BERT (CPU, no downloads)."""
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src import er_crossenc as ce  # noqa: E402
from src.er_synthetic import make_split  # noqa: E402

COLS = ["entity_id", "business_name", "business_address", "country"]


@pytest.fixture(autouse=True)
def _dirs(tmp_path, monkeypatch):
    """Keep heartbeats and data caches inside tmp_path."""
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))


@pytest.fixture(scope="module")
def synth():
    """Synthetic sources (S1, S2, S3) + a stage-1-like candidate table with labels and probs."""
    s1, s2, s3, gt = make_split(["US", "India", "France"], 40, random.Random(0), "t")
    s1, s2, s3 = (pd.DataFrame(x, columns=COLS) for x in (s1, s2, s3))
    rng = np.random.default_rng(0)
    right_ids = pd.concat([s2, s3]).entity_id.to_numpy()
    rows = []
    for sid, matches in gt.items():
        for c in matches:
            rows.append((sid, c, 0.6 + 0.4 * rng.random(), True))
        for c in rng.choice(right_ids, 4, replace=False):
            if c not in matches:
                rows.append((sid, c, 0.7 * rng.random(), False))
    chunks = pd.DataFrame(rows, columns=["s1_id", "cand_id", "prob", "y"])
    chunks["prob"] = chunks.prob.astype(np.float32)
    left = s1.set_index("entity_id")[["business_name", "business_address"]]
    right = pd.concat([s2, s3]).set_index("entity_id")[["business_name", "business_address"]]
    return {"s1": s1, "s2": s2, "s3": s3, "chunks": chunks, "left": left, "right": right}


def _words(text: str) -> list[str]:
    """Lowercase, accent-stripped word/punctuation tokens (mirrors BERT basic tokenization closely enough)."""
    t = "".join(ch for ch in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(ch) != "Mn")
    return re.findall(r"\w+|[^\w\s]", t)


@pytest.fixture(scope="module")
def tiny_model(synth, tmp_path_factory):
    """Randomly initialised 2-layer BERT + WordPiece vocab built from the synthetic text, saved to disk."""
    d = tmp_path_factory.mktemp("tiny_bert")
    texts = [ce.record_text(n, a) for df in (synth["s1"], synth["s2"], synth["s3"])
             for n, a in zip(df.business_name, df.business_address)]
    words = sorted({w for t in texts for w in _words(t)})
    chars = sorted({c for w in words for c in w})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + [c for c in chars if c not in words]
    vocab += ["##" + c for c in chars]
    tok = transformers.BertTokenizer(vocab={w: i for i, w in enumerate(vocab)})
    torch.manual_seed(0)
    cfg = transformers.BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                  intermediate_size=64, max_position_embeddings=128, num_labels=1)
    transformers.BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    assert tok.convert_tokens_to_ids("[UNK]") not in tok(texts[0])["input_ids"]   # vocab covers the text
    return str(d)


def _state(model_dir):
    """Model weights of a saved cross-encoder as a dict of tensors."""
    m = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir)
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_record_text():
    assert ce.record_text("Acme Foods", "12 Main St, Austin") == "Acme Foods | 12 Main St, Austin"
    assert ce.record_text("श्री गणेश", "") == "श्री गणेश | "


def test_sample_training_pairs_ratio_and_hardness():
    rng = np.random.default_rng(1)
    n = 20_000
    y = rng.random(n) < 0.1
    prob = np.where(y, 0.5 + 0.5 * rng.random(n), rng.random(n) ** 3).astype(np.float32)
    ch = pd.DataFrame({"s1_id": [f"S1-{i // 10}" for i in range(n)], "cand_id": [f"S2-{i}" for i in range(n)],
                       "prob": prob, "y": y, "other_feat": 1.0})
    p = ce.sample_training_pairs(ch, 2000, pos_frac=0.4, seed=0)
    assert list(p.columns) == ["s1_id", "cand_id", "y"] and len(p) == 2000
    assert p.y.sum() == 800 and p.y.dtype == bool
    assert not p.duplicated(["s1_id", "cand_id"]).any()
    neg_prob = ch.set_index("cand_id").prob.reindex(p.cand_id[~p.y]).to_numpy()
    assert neg_prob.mean() > 1.5 * prob[~y].mean()                  # hard negatives favoured
    truth = set(ch.cand_id[ch.y])
    assert all((c in truth) == lab for c, lab in zip(p.cand_id, p.y))
    pd.testing.assert_frame_equal(p, ce.sample_training_pairs(ch, 2000, pos_frac=0.4, seed=0))  # deterministic
    few = ce.sample_training_pairs(ch.iloc[:200], 150, pos_frac=0.9, seed=0)      # too few positives -> fill
    assert len(few) == 150 and few.y.sum() == ch.y.iloc[:200].sum()


def test_train_and_score_end_to_end(synth, tiny_model, tmp_path):
    pairs = ce.sample_training_pairs(synth["chunks"], 120, seed=0)
    out = ce.train(pairs, synth["left"], synth["right"], tmp_path / "ce", model_name=tiny_model, epochs=2,
                   batch=16, lr=1e-3, max_len=48, device="cpu", ckpt_every=3)
    assert (out / "config.json").exists() and (out / "tokenizer_config.json").exists()
    meta = json.loads((out / "train_meta.json").read_text())
    assert meta["steps"] == 2 * int(np.ceil(len(pairs) / 16)) and meta["resumed_from_step"] is None
    assert json.loads((tmp_path / "runs" / "crossenc-ce" / "heartbeat.json").read_text())["status"] == "completed"

    test_pairs = synth["chunks"][["s1_id", "cand_id"]]
    p = ce.score(out, test_pairs, synth["left"], synth["right"], batch=512, max_len=48, device="cpu")
    assert p.dtype == np.float32 and p.shape == (len(test_pairs),)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    rev = test_pairs.iloc[::-1].reset_index(drop=True)
    p_rev = ce.score(out, rev, synth["left"], synth["right"], batch=7, max_len=48, device="cpu")
    np.testing.assert_allclose(p_rev[::-1], p, atol=1e-5)             # order kept, batching/padding-invariant
    assert len(ce.score(out, test_pairs.iloc[:0], synth["left"], synth["right"], device="cpu")) == 0
    with pytest.raises(KeyError):
        ce.score(out, pd.DataFrame({"s1_id": ["nope"], "cand_id": [test_pairs.cand_id.iloc[0]]}),
                 synth["left"], synth["right"], device="cpu")


def test_resume_matches_uninterrupted_run(synth, tiny_model, tmp_path):
    pairs = ce.sample_training_pairs(synth["chunks"], 96, seed=1)
    kw = dict(model_name=tiny_model, epochs=2, batch=16, lr=1e-3, max_len=48, device="cpu", ckpt_every=2, seed=3)
    full = ce.train(pairs, synth["left"], synth["right"], tmp_path / "full", **kw)

    part = tmp_path / "part"
    ce.train(pairs, synth["left"], synth["right"], part, max_steps=5, **kw)        # "crash" mid epoch 0
    assert (part / "ckpt" / "last.pt").exists() and not (part / "train_meta.json").exists()
    ce.train(pairs, synth["left"], synth["right"], part, **kw)                      # resume
    assert json.loads((part / "train_meta.json").read_text())["resumed_from_step"] == 5
    a, b = _state(full), _state(part)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-6)

    mtime = (part / "train_meta.json").stat().st_mtime
    ce.train(pairs, synth["left"], synth["right"], part, **kw)                      # finished -> no-op
    assert (part / "train_meta.json").stat().st_mtime == mtime

    other = tmp_path / "other"
    ce.train(pairs, synth["left"], synth["right"], other, max_steps=2, **kw)
    with pytest.raises(ValueError, match="different pairs"):
        ce.train(pairs.iloc[1:], synth["left"], synth["right"], other, **kw)


def test_cli_train_and_score(synth, tiny_model, tmp_path):
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    for k in (1, 2, 3):
        synth[f"s{k}"].to_parquet(cache / f"train_s{k}.parquet", index=False)
    chunks_dir = tmp_path / "train_chunks"
    chunks_dir.mkdir()
    ch = synth["chunks"].assign(other_feat=np.float32(0.5))
    half = ch.s1_id.isin(ch.s1_id.unique()[:20])
    ch[half].to_parquet(chunks_dir / "000.parquet", index=False)
    ch[~half].to_parquet(chunks_dir / "001.parquet", index=False)

    model_dir = tmp_path / "ce_model"
    common = ["--device", "cpu", "--max-len", "48"]
    ce.main(["train", "--chunks", str(chunks_dir), "--out", str(model_dir), "--n", "100", "--split", "train",
             "--model-name", tiny_model, "--batch", "16", "--lr", "1e-3", *common])
    sampled = pd.read_parquet(model_dir / "train_pairs.parquet")
    assert len(sampled) == 100 and sampled.y.mean() == pytest.approx(0.4, abs=0.01)
    assert (model_dir / "train_meta.json").exists()

    out = tmp_path / "train_ce"
    ce.main(["score", "--model", str(model_dir), "--chunks", str(chunks_dir), "--split", "train", "--out", str(out),
             *common])
    for name in ("000.parquet", "001.parquet"):
        src, got = pd.read_parquet(chunks_dir / name), pd.read_parquet(out / name)
        assert list(got.columns) == ["s1_id", "cand_id", "ce_score"]
        assert got.s1_id.tolist() == src.s1_id.tolist() and got.cand_id.tolist() == src.cand_id.tolist()
        assert got.ce_score.between(0, 1).all()

    (out / "001.parquet").unlink()                                                  # resumable: redo only 001
    mtime0 = (out / "000.parquet").stat().st_mtime
    ce.main(["score", "--model", str(model_dir), "--chunks", str(chunks_dir), "--split", "train", "--out", str(out),
             *common])
    assert (out / "001.parquet").exists() and (out / "000.parquet").stat().st_mtime == mtime0

    # --keep-prob / --keep-ce-dir: only final candidate rows (prob >= t OR filter CE >= c) scored, rows stay aligned
    filt = tmp_path / "filt" / "train_ce"
    filt.mkdir(parents=True)
    for name in ("000.parquet", "001.parquet"):
        s = pd.read_parquet(out / name)
        s.assign(ce_score=np.where(np.arange(len(s)) % 3 == 0, 0.9, 0.001).astype(np.float32)).to_parquet(filt / name)
    kept = tmp_path / "kept_ce"
    ce.main(["score", "--model", str(model_dir), "--chunks", str(chunks_dir), "--split", "train", "--out", str(kept),
             "--keep-prob", "0.5", "--keep-ce-dir", str(tmp_path / "filt"), "--keep-ce", "0.01", *common])
    for name in ("000.parquet", "001.parquet"):
        src, full, got = pd.read_parquet(chunks_dir / name), pd.read_parquet(out / name), pd.read_parquet(kept / name)
        want = (src.prob.to_numpy() >= 0.5) | (np.arange(len(src)) % 3 == 0)
        assert got.cand_id.tolist() == src.cand_id.tolist()
        assert got.ce_score.notna().to_numpy().tolist() == want.tolist()
        np.testing.assert_allclose(got.ce_score[want], full.ce_score[want], atol=1e-5)
