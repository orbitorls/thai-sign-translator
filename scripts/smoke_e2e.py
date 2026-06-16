"""End-to-end smoke test for the Thai Sign Language translation system.

Runs entirely on CPU with no network downloads.

Checks:
  1. Normalization pipeline   : (10, 543, 3) → normalize_sequence() → (10, 312)
  2. Model forward pass       : tiny PoseToTextT5 (monkeypatched) → finite loss
  3. Video pipeline translation: (10, 543, 3) → translate_video() → str
  4. API smoke via TestClient : POST /translate-video → 200 + sentence str

Run:
    python scripts/smoke_e2e.py

Exits 0 on success, non-zero on failure.
"""
from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Ensure src/ is on sys.path when running from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import math
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import MT5Config, MT5ForConditionalGeneration


# ---------------------------------------------------------------------------
# Shared stub translator
# ---------------------------------------------------------------------------

class _StubTranslator:
    """Returns a fixed prediction for any input — no model weights needed."""

    def translate(self, features: np.ndarray):
        return SimpleNamespace(sentence="สวัสดี", token_ids=[1, 2], score=0.9)


# ---------------------------------------------------------------------------
# Check 1 — Normalization pipeline
# ---------------------------------------------------------------------------

def check_normalization() -> None:
    from tsl.features.normalize import normalize_sequence

    raw = np.random.randn(10, 543, 3).astype(np.float32)
    out = normalize_sequence(raw)

    assert out.shape == (10, 312), (
        f"Expected shape (10, 312), got {out.shape}"
    )
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    assert np.isfinite(out).all(), "normalize_sequence output contains non-finite values"
    print("✓ normalization pipeline")


# ---------------------------------------------------------------------------
# Check 2 — Model forward pass (tiny patched PoseToTextT5)
# ---------------------------------------------------------------------------

def check_model_forward() -> None:
    from tsl.models.pose_t5 import PoseToTextT5

    # Build a tiny MT5 config that avoids any network download.
    # d_model=32, num_heads=2 → nhead=2 (32 % 2 == 0 ✓)
    tiny_config = MT5Config(
        d_model=32,
        num_heads=2,
        num_layers=1,
        num_decoder_layers=1,
        d_ff=64,
        vocab_size=250112,
    )
    tiny_t5 = MT5ForConditionalGeneration(tiny_config)

    # Monkeypatch from_pretrained so PoseToTextT5.__init__ gets the tiny model
    # without touching disk or network.
    with patch.object(
        MT5ForConditionalGeneration,
        "from_pretrained",
        return_value=tiny_t5,
    ):
        model = PoseToTextT5(
            input_dim=312,
            num_encoder_layers=1,
            encoder_dropout=0.0,
            downsample_factor=1,   # keep T=10 → 10 after downsample (no padding waste)
            base_model_name="dummy-no-download",
        )

    model.eval()

    # Minimal batch: 1 sample, 10 frames, 312 dims
    src = torch.randn(1, 10, 312)
    src_lengths = torch.tensor([10])
    labels = torch.randint(0, 100, (1, 4))  # (B, T_tgt)

    with torch.no_grad():
        out = model(src, src_lengths, labels=labels)

    loss = out.loss
    assert loss is not None, "Expected a loss tensor, got None"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    print("✓ model forward pass")


# ---------------------------------------------------------------------------
# Check 3 — Video pipeline translation
# ---------------------------------------------------------------------------

def check_video_pipeline() -> None:
    from tsl.inference.video_pipeline import translate_video

    raw = np.random.randn(10, 543, 3).astype(np.float32)
    stub = _StubTranslator()

    result = translate_video(raw, stub)

    assert isinstance(result, str), f"Expected str, got {type(result).__name__!r}"
    assert len(result) > 0, "translate_video returned an empty string"
    print("✓ video pipeline translation")


# ---------------------------------------------------------------------------
# Check 4 — API smoke via FastAPI TestClient
# ---------------------------------------------------------------------------

def check_api_smoke() -> None:
    from fastapi.testclient import TestClient
    from tsl.api.app import app, get_active_translator

    stub = _StubTranslator()
    app.dependency_overrides[get_active_translator] = lambda: stub

    try:
        client = TestClient(app)

        # 5 frames of (543, 3) raw landmarks → should normalize + translate
        frames = np.zeros((5, 543, 3), dtype=np.float32).tolist()
        resp = client.post(
            "/translate-video",
            json={"frames": frames, "feature_schema": "raw_mediapipe_543x3"},
        )

        assert resp.status_code == 200, (
            f"Expected 200, got {resp.status_code}: {resp.text}"
        )
        body = resp.json()
        assert "sentence" in body, f"Response missing 'sentence' key: {body}"
        assert isinstance(body["sentence"], str), (
            f"'sentence' is not a string: {body['sentence']!r}"
        )
        assert len(body["sentence"]) > 0, "API returned empty sentence"
        print("✓ API smoke via TestClient")
    finally:
        app.dependency_overrides.pop(get_active_translator, None)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    checks = [
        ("normalization pipeline", check_normalization),
        ("model forward pass",     check_model_forward),
        ("video pipeline",         check_video_pipeline),
        ("API smoke",              check_api_smoke),
    ]

    failed = False
    for name, fn in checks:
        try:
            fn()
        except Exception as exc:
            print(f"✗ {name}")
            traceback.print_exc()
            failed = True

    if failed:
        print("\nOne or more smoke checks FAILED.")
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
