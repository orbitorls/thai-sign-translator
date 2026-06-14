import torch
import torch.nn.functional as F

from tsl.models.slt import SignToTextTransformer


def _make_model(
    input_dim: int = 12,
    vocab_size: int = 20,
    d_model: int = 16,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dim_feedforward: int = 32,
    dropout: float = 0.0,
) -> SignToTextTransformer:
    return SignToTextTransformer(
        input_dim=input_dim,
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


def test_forward_shape():
    torch.manual_seed(0)
    model = _make_model(input_dim=162, vocab_size=100, d_model=128, nhead=4)
    model.eval()
    src = torch.randn(2, 10, 162)
    src_lengths = torch.tensor([10, 10])
    tgt = torch.randint(0, 100, (2, 7))
    logits = model(src, src_lengths, tgt)
    assert logits.shape == (2, 7, 100)
    assert torch.isfinite(logits).all()


def test_forward_with_padding_mask():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 8, 12)
    src_lengths = torch.tensor([8, 5])
    tgt = torch.randint(0, 20, (2, 4))
    logits = model(src, src_lengths, tgt)
    assert logits.shape == (2, 4, 20)
    assert torch.isfinite(logits).all()


def test_loss_backprop():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    src = torch.randn(2, 8, 12)
    src_lengths = torch.tensor([8, 5])
    tgt = torch.randint(0, 20, (2, 6))
    target = torch.randint(0, 20, (2, 6))
    logits = model(src, src_lengths, tgt)
    loss = F.cross_entropy(logits.reshape(-1, 20), target.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).any()
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad


def test_greedy_decode_shape():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 10, 12)
    src_lengths = torch.tensor([10, 10])
    out = model.greedy_decode(src, src_lengths, bos_id=0, eos_id=1, max_len=15)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert 1 <= out.shape[1] <= 15
    assert (out[:, 0] == 0).all()


def test_greedy_decode_stops_at_eos():
    torch.manual_seed(0)
    model = _make_model(input_dim=8, vocab_size=5, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 6, 8)
    src_lengths = torch.tensor([6, 6])
    out = model.greedy_decode(src, src_lengths, bos_id=0, eos_id=2, max_len=20)
    assert (out[:, -1] == 2).any().item()


def test_causal_mask_works():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 6, 12)
    src_lengths = torch.tensor([6, 6])
    tgt = torch.randint(0, 20, (2, 5))
    T_tgt = tgt.size(1)
    explicit_mask = torch.triu(torch.full((T_tgt, T_tgt), float("-inf")), diagonal=1)
    logits = model(src, src_lengths, tgt, tgt_mask=explicit_mask)
    assert logits.shape == (2, 5, 20)
    assert torch.isfinite(logits).all()


def test_does_not_pool_to_single_vector():
    torch.manual_seed(0)
    model = _make_model(input_dim=12, vocab_size=20, d_model=16, nhead=4)
    model.eval()
    src = torch.randn(2, 10, 12)
    src_lengths = torch.tensor([10, 10])
    memory = model.encode(src, src_lengths)
    assert memory.ndim == 3
    assert memory.shape == (2, 10, 16)
