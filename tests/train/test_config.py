from __future__ import annotations

from tsl.train.config import resolve_config


def test_small_preset():
    cfg = resolve_config("small", input_dim=162)
    assert cfg.d_model == 64
    assert cfg.nhead == 4
    assert cfg.num_encoder_layers == 2
    assert cfg.num_decoder_layers == 2
    assert cfg.dim_feedforward == 128
    assert cfg.input_dim == 162


def test_base_preset():
    cfg = resolve_config("base", input_dim=411)
    assert cfg.d_model == 256
    assert cfg.nhead == 8
    assert cfg.num_encoder_layers == 4
    assert cfg.num_decoder_layers == 4
    assert cfg.dim_feedforward == 1024
    assert cfg.input_dim == 411


def test_large_preset():
    cfg = resolve_config("large", input_dim=162)
    assert cfg.d_model == 512
    assert cfg.nhead == 8
    assert cfg.num_encoder_layers == 6
    assert cfg.num_decoder_layers == 6
    assert cfg.dim_feedforward == 2048
    assert cfg.max_pos_len == 2048


def test_input_dim_carries_through():
    cfg = resolve_config("small", input_dim=99)
    assert cfg.input_dim == 99
