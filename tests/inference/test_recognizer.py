import numpy as np
import torch
import pytest
from tsl.models.encoder import LandmarkEncoder
from tsl.registry.prototype_store import PrototypeStore
from tsl.inference.recognizer import Recognizer


def _make_encoder(input_dim=12):
    torch.manual_seed(0)
    enc = LandmarkEncoder(input_dim=input_dim, emb_dim=16, d_model=32, nhead=2, num_layers=1)
    enc.eval()
    return enc


def _clip(T, D, base, seed):
    rng = np.random.default_rng(seed)
    return (base + 0.01 * rng.standard_normal((T, D))).astype(np.float32)


def _store_with_two_signs(D=12):
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    clips_a = [_clip(6, D, np.full(D, -1.0, np.float32), s) for s in range(3)]
    clips_b = [_clip(6, D, np.full(D, +1.0, np.float32), s + 100) for s in range(3)]
    store.add_sign("alpha", clips_a)
    store.add_sign("beta", clips_b)
    return store


def test_recognize_returns_contract_dict():
    D = 12
    store = _store_with_two_signs(D)
    rec = Recognizer(store)
    query_like_a = _clip(6, D, np.full(D, -1.0, np.float32), 999)
    out = rec.recognize(query_like_a)
    assert set(out.keys()) == {"word", "score", "topk"}
    assert out["word"] == "alpha"
    assert isinstance(out["score"], float)
    assert isinstance(out["topk"], list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in out["topk"])
    assert {w for w, _ in out["topk"]} == {"alpha", "beta"}


def test_recognize_topk_sorted_descending_and_top_matches_word():
    D = 12
    store = _store_with_two_signs(D)
    rec = Recognizer(store)
    query_like_b = _clip(6, D, np.full(D, +1.0, np.float32), 777)
    out = rec.recognize(query_like_b)
    scores = [s for _, s in out["topk"]]
    assert scores == sorted(scores, reverse=True)
    assert out["topk"][0][0] == out["word"]
    assert out["score"] == out["topk"][0][1]


def test_recognize_empty_store_raises():
    enc = _make_encoder(12)
    rec = Recognizer(PrototypeStore(enc))
    with pytest.raises(ValueError):
        rec.recognize(_clip(5, 12, 0.0, 1))
