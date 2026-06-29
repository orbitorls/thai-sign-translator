import numpy as np
import torch
import tsl.api.app as appmod


def test_build_encoder_is_deterministic_without_weights(monkeypatch):
    # Force the no-weights branch.
    monkeypatch.setattr(appmod.config, "ENCODER_WEIGHTS_PATH", None, raising=False)
    enc_a = appmod._build_encoder()
    enc_b = appmod._build_encoder()

    # Same random seed → identical parameters → identical embeddings.
    seq = np.zeros((4, len(appmod.SELECTED_LANDMARKS) * 3), dtype=np.float32)
    x = torch.from_numpy(seq).unsqueeze(0)
    lengths = torch.tensor([4], dtype=torch.long)
    with torch.no_grad():
        emb_a = enc_a(x, lengths)
        emb_b = enc_b(x, lengths)
    assert torch.allclose(emb_a, emb_b)
