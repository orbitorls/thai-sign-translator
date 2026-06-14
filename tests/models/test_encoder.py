import torch

from tsl.models.encoder import LandmarkEncoder


def test_encoder_output_shape():
    torch.manual_seed(0)
    enc = LandmarkEncoder(input_dim=12, emb_dim=8, d_model=16, nhead=4, num_layers=2)
    enc.eval()
    B, T, D = 3, 5, 12
    x = torch.randn(B, T, D)
    lengths = torch.tensor([5, 5, 5])
    out = enc(x, lengths)
    assert out.shape == (B, 8)
    assert torch.isfinite(out).all()


def test_encoder_pad_invariance():
    torch.manual_seed(0)
    enc = LandmarkEncoder(input_dim=12, emb_dim=8, d_model=16, nhead=4, num_layers=2)
    enc.eval()
    D = 12
    real = torch.randn(1, 4, D)
    lengths = torch.tensor([4])
    out_tight = enc(real, lengths)
    padded = torch.cat([real, torch.randn(1, 3, D)], dim=1)
    out_padded = enc(padded, lengths)
    assert torch.allclose(out_tight, out_padded, atol=1e-5)


def test_encoder_batch_independent_lengths():
    torch.manual_seed(0)
    enc = LandmarkEncoder(input_dim=12, emb_dim=8, d_model=16, nhead=4, num_layers=2)
    enc.eval()
    D = 12
    x = torch.randn(2, 6, D)
    lengths = torch.tensor([6, 2])
    out = enc(x, lengths)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
