import numpy as np
import torch
import pytest
from tsl.models.encoder import LandmarkEncoder
from tsl.registry.prototype_store import PrototypeStore


def _make_encoder(input_dim=12):
    torch.manual_seed(0)
    enc = LandmarkEncoder(input_dim=input_dim, emb_dim=16, d_model=32, nhead=2, num_layers=1)
    enc.eval()
    return enc


def _clip(T, D, base, seed):
    rng = np.random.default_rng(seed)
    return (base + 0.01 * rng.standard_normal((T, D))).astype(np.float32)


def test_add_sign_stores_prototype_and_names():
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    clips_a = [_clip(5, D, 0.0, s) for s in range(3)]
    store.add_sign("alpha", clips_a)
    assert store.names() == ["alpha"]


def test_predict_returns_closer_sign():
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    clips_a = [_clip(6, D, np.full(D, -1.0, np.float32), s) for s in range(3)]
    clips_b = [_clip(6, D, np.full(D, +1.0, np.float32), s + 100) for s in range(3)]
    store.add_sign("alpha", clips_a)
    store.add_sign("beta", clips_b)
    query_like_a = _clip(6, D, np.full(D, -1.0, np.float32), 999)
    word, score = store.predict(query_like_a)
    assert word == "alpha"
    assert isinstance(score, float)


def test_add_sign_uses_no_gradient():
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    clips_a = [_clip(5, D, 0.0, s) for s in range(2)]
    store.add_sign("alpha", clips_a)
    proto = store._prototypes["alpha"]
    assert isinstance(proto, torch.Tensor)
    assert proto.requires_grad is False
    assert proto.grad_fn is None


def test_remove_sign():
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    store.add_sign("alpha", [_clip(5, D, 0.0, 1)])
    store.add_sign("beta", [_clip(5, D, 1.0, 2)])
    store.remove_sign("alpha")
    assert store.names() == ["beta"]


def test_save_load_round_trip(tmp_path):
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    clips_a = [_clip(6, D, np.full(D, -1.0, np.float32), s) for s in range(3)]
    clips_b = [_clip(6, D, np.full(D, +1.0, np.float32), s + 100) for s in range(3)]
    store.add_sign("alpha", clips_a)
    store.add_sign("beta", clips_b)
    path = tmp_path / "store.pt"
    store.save(str(path))
    enc2 = _make_encoder(D)
    loaded = PrototypeStore.load(str(path), enc2)
    assert loaded.names() == ["alpha", "beta"]
    for name in ("alpha", "beta"):
        assert torch.allclose(loaded._prototypes[name], store._prototypes[name])


def test_predict_raises_when_empty():
    D = 12
    enc = _make_encoder(D)
    store = PrototypeStore(enc)
    with pytest.raises(ValueError):
        store.predict(_clip(5, D, 0.0, 1))
