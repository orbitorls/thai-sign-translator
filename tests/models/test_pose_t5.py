"""Tests for PoseToTextT5 model.

All tests use a tiny mT5 config so no network download is needed.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from transformers import MT5Config, MT5ForConditionalGeneration


# ---------------------------------------------------------------------------
# Tiny MT5 fixture (no network access required)
# ---------------------------------------------------------------------------

TINY_CONFIG = MT5Config(
    d_model=32,
    num_heads=2,
    num_layers=1,
    d_ff=64,
    d_kv=16,
    vocab_size=250112,
    decoder_start_token_id=0,
    eos_token_id=1,
    pad_token_id=0,
)


def _tiny_model_factory(*args, **kwargs) -> MT5ForConditionalGeneration:
    """Return a tiny MT5 model without downloading anything."""
    return MT5ForConditionalGeneration(TINY_CONFIG)


@pytest.fixture()
def patch_mt5(monkeypatch):
    """Patch MT5ForConditionalGeneration.from_pretrained in pose_t5 module."""
    monkeypatch.setattr(
        "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
        _tiny_model_factory,
    )


@pytest.fixture()
def model(patch_mt5):
    """A PoseToTextT5 built with the tiny MT5 config."""
    # Import after patching so the patch is already active
    from tsl.models.pose_t5 import PoseToTextT5

    return PoseToTextT5(
        input_dim=312,
        num_encoder_layers=2,
        encoder_dropout=0.1,
        downsample_factor=4,
        base_model_name="google/mt5-small",
    )


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _random_batch(B: int = 2, T: int = 40, D: int = 312):
    src = torch.randn(B, T, D)
    src_lengths = torch.tensor([T, T - 8], dtype=torch.long)
    return src, src_lengths


def _random_labels(B: int = 2, T_tgt: int = 8, vocab_size: int = 250112):
    labels = torch.randint(0, vocab_size, (B, T_tgt), dtype=torch.long)
    # Mask last token of second sequence to simulate padding
    labels[1, -1] = -100
    return labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForward:
    def test_loss_is_finite(self, model):
        src, src_lengths = _random_batch()
        labels = _random_labels()
        out = model(src, src_lengths, labels=labels)
        assert out.loss is not None
        assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss}"

    def test_logits_shape(self, model):
        B, T_tgt = 2, 8
        src, src_lengths = _random_batch(B=B, T=40)
        labels = _random_labels(B=B, T_tgt=T_tgt)
        out = model(src, src_lengths, labels=labels)
        # logits: (B, T_tgt, vocab_size)
        assert out.logits.shape[0] == B
        assert out.logits.shape[1] == T_tgt
        assert out.logits.shape[2] == TINY_CONFIG.vocab_size

    def test_no_labels_returns_logits(self, model):
        """Forward without labels should still return logits (teacher-forced with decoder_input_ids)."""
        src, src_lengths = _random_batch()
        # Pass decoder_input_ids so T5 can produce logits without labels
        decoder_input_ids = torch.zeros(2, 4, dtype=torch.long)
        memory, attn_mask = model._build_pose_embeds(src, src_lengths)
        out = model.t5_model(
            inputs_embeds=memory,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
        )
        assert out.logits is not None
        assert out.loss is None


class TestTemporalDownsampling:
    def test_downsampling_reduces_length(self, model):
        """T=40 with factor=4 should give T_ds=10."""
        src, src_lengths = _random_batch(T=40)
        x_ds, ds_lengths = model._downsample(src, src_lengths)
        assert x_ds.size(1) == 10  # 40 / 4

    def test_downsampling_non_multiple_pads(self, model):
        """T=41 with factor=4 should give T_ds=ceil(41/4)=11."""
        src = torch.randn(2, 41, 312)
        src_lengths = torch.tensor([41, 38], dtype=torch.long)
        x_ds, ds_lengths = model._downsample(src, src_lengths)
        assert x_ds.size(1) == 11  # ceil(41/4)

    def test_ds_lengths_ceiling(self, model):
        """ds_lengths should be ceiling division of src_lengths."""
        src_lengths = torch.tensor([40, 33], dtype=torch.long)
        src = torch.randn(2, 40, 312)
        _, ds_lengths = model._downsample(src, src_lengths)
        expected = torch.tensor([10, 9], dtype=torch.long)  # ceil(40/4)=10, ceil(33/4)=9
        assert torch.all(ds_lengths == expected.to(ds_lengths.device))

    def test_custom_downsample_factor(self, patch_mt5):
        from tsl.models.pose_t5 import PoseToTextT5
        model = PoseToTextT5(downsample_factor=8)
        src = torch.randn(2, 64, 312)
        src_lengths = torch.tensor([64, 48], dtype=torch.long)
        x_ds, ds_lengths = model._downsample(src, src_lengths)
        assert x_ds.size(1) == 8  # 64 / 8
        assert ds_lengths[0].item() == 8
        assert ds_lengths[1].item() == 6  # ceil(48/8)


class TestPoseEmbeds:
    def test_attention_mask_dtype_and_range(self, model):
        src, src_lengths = _random_batch(T=16)
        memory, attn_mask = model._build_pose_embeds(src, src_lengths)
        assert attn_mask.dtype == torch.long
        assert attn_mask.min().item() >= 0
        assert attn_mask.max().item() <= 1

    def test_memory_shape(self, model):
        B, T = 3, 24
        src = torch.randn(B, T, 312)
        src_lengths = torch.tensor([24, 20, 16], dtype=torch.long)
        memory, attn_mask = model._build_pose_embeds(src, src_lengths)
        T_ds = T // 4  # 6
        proj_dim = TINY_CONFIG.d_model  # 32
        assert memory.shape == (B, T_ds, proj_dim)
        assert attn_mask.shape == (B, T_ds)

    def test_padding_reflected_in_attn_mask(self, model):
        """The shorter sequence (src_lengths[1] < src_lengths[0]) should have fewer 1s."""
        B, T = 2, 20
        src = torch.randn(B, T, 312)
        src_lengths = torch.tensor([20, 8], dtype=torch.long)
        _, attn_mask = model._build_pose_embeds(src, src_lengths)
        ones_0 = attn_mask[0].sum().item()
        ones_1 = attn_mask[1].sum().item()
        assert ones_0 > ones_1


class TestGenerate:
    def test_generate_returns_token_ids(self, model):
        src, src_lengths = _random_batch(B=2, T=16)
        out = model.generate(src, src_lengths, max_new_tokens=5)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2
        assert out.shape[0] == 2

    def test_generate_is_no_grad(self, model):
        """generate() must not accumulate gradients."""
        src, src_lengths = _random_batch(T=16)
        out = model.generate(src, src_lengths, max_new_tokens=3)
        assert not out.requires_grad


class TestSaveLoadRoundTrip:
    def test_round_trip(self, monkeypatch):
        """save_pretrained / from_pretrained preserves pose encoder weights."""
        # Build tiny model with real tiny weights (no network)
        from tsl.models.pose_t5 import PoseToTextT5

        # Patch for the initial model construction
        monkeypatch.setattr(
            "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
            _tiny_model_factory,
        )

        model = PoseToTextT5(
            input_dim=312,
            num_encoder_layers=1,
            encoder_dropout=0.0,
            downsample_factor=4,
            base_model_name="google/mt5-small",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, "pose_encoder.pt"))
            assert os.path.exists(os.path.join(tmpdir, "pose_t5_config.json"))

            # Check config contents
            with open(os.path.join(tmpdir, "pose_t5_config.json")) as fh:
                cfg = json.load(fh)
            assert cfg["input_dim"] == 312
            assert cfg["downsample_factor"] == 4

            # Patch from_pretrained again for the load call
            monkeypatch.setattr(
                "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
                _tiny_model_factory,
            )

            loaded = PoseToTextT5.from_pretrained(tmpdir, device="cpu")

        # Compare pose encoder weights
        orig_state = model.pose_encoder_state()
        load_state = loaded.pose_encoder_state()

        for sub_key in ("input_proj", "pos_enc", "transformer_encoder"):
            for param_key, orig_val in orig_state[sub_key].items():
                loaded_val = load_state[sub_key][param_key]
                assert torch.allclose(orig_val.cpu(), loaded_val.cpu()), (
                    f"Mismatch in {sub_key}.{param_key}"
                )

    def test_from_pretrained_sets_eval_mode(self, monkeypatch):
        from tsl.models.pose_t5 import PoseToTextT5

        monkeypatch.setattr(
            "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
            _tiny_model_factory,
        )
        model = PoseToTextT5(num_encoder_layers=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            monkeypatch.setattr(
                "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
                _tiny_model_factory,
            )
            loaded = PoseToTextT5.from_pretrained(tmpdir)

        assert not loaded.training, "Model should be in eval mode after from_pretrained"


class TestPoseEncoderState:
    def test_pose_encoder_state_keys(self, model):
        state = model.pose_encoder_state()
        assert set(state.keys()) == {"input_proj", "pos_enc", "transformer_encoder"}

    def test_t5_weights_not_in_pose_state(self, model):
        """pose_encoder_state should not contain T5 parameters."""
        state = model.pose_encoder_state()
        # All sub-dicts should only contain pose-front-end parameters
        all_keys = []
        for sub in state.values():
            all_keys.extend(sub.keys())
        # T5 params have names like 'encoder.block.0...' — shouldn't appear
        for key in all_keys:
            assert "encoder.block" not in key
            assert "decoder.block" not in key
