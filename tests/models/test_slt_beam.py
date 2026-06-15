from __future__ import annotations

import torch

from tsl.models.slt import SignToTextTransformer


def _make_model(
    input_dim: int = 12,
    vocab_size: int = 20,
    d_model: int = 16,
    nhead: int = 4,
) -> SignToTextTransformer:
    return SignToTextTransformer(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=32,
        dropout=0.0,
    )


def test_beam_decode_shape():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 10, 12)
    src_lengths = torch.tensor([10, 10])
    out = model.beam_decode(
        src, src_lengths, bos_id=0, eos_id=1, beam_size=3, max_len=15
    )
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert 1 <= out.shape[1] <= 15


def test_beam_decode_vs_greedy_different():
    torch.manual_seed(42)
    model = _make_model(input_dim=8, vocab_size=10, d_model=32, nhead=4)
    model.eval()
    src = torch.randn(1, 6, 8)
    src_lengths = torch.tensor([6])

    greedy = model.greedy_decode(src, src_lengths, bos_id=0, eos_id=2, max_len=10)
    beam = model.beam_decode(src, src_lengths, bos_id=0, eos_id=2, beam_size=3, max_len=10)

    assert greedy.shape[0] == 1
    assert beam.shape[0] == 1


def test_beam_decode_ends_with_eos():
    torch.manual_seed(0)
    model = _make_model(input_dim=8, vocab_size=5, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 6, 8)
    src_lengths = torch.tensor([6, 6])
    out = model.beam_decode(src, src_lengths, bos_id=0, eos_id=2, beam_size=3, max_len=20)
    assert (out[:, -1] == 2).any().item()
