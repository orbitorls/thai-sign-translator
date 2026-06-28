"""Minimal PoseT5 runtime bundle wrapper for repeatable inference paths."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tsl.inference.pose_t5_translator import PoseT5Translator

_REQUIRED_FILES = (
    "model.safetensors",
    "pose_encoder.pt",
    "pose_t5_config.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class BundleValidationError(ValueError):
    """Raised when a runtime export is missing required files or metadata."""


@dataclass(frozen=True)
class BundleMetadata:
    model_dir: str
    model_id: str
    feature_dim: int
    num_encoder_layers: int
    encoder_dropout: float
    downsample_factor: int
    base_model_name: str
    decode_config: dict[str, int | float | None]
    missing_files: list[str]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_decode_config(model_dir: Path) -> dict[str, int | float | None]:
    verified_eval = model_dir / "verified_eval.json"
    if verified_eval.is_file():
        payload = _load_json(verified_eval)
        decoding = payload.get("decoding")
        if isinstance(decoding, dict):
            return {
                "max_new_tokens": decoding.get("max_new_tokens", PoseT5Translator.DEFAULT_MAX_NEW_TOKENS),
                "beam_size": decoding.get("beam_size", PoseT5Translator.DEFAULT_BEAM_SIZE),
                "no_repeat_ngram_size": decoding.get(
                    "no_repeat_ngram_size",
                    PoseT5Translator.DEFAULT_NO_REPEAT_NGRAM_SIZE,
                ),
                "repetition_penalty": decoding.get(
                    "repetition_penalty",
                    PoseT5Translator.DEFAULT_REPETITION_PENALTY,
                ),
                "length_penalty": decoding.get(
                    "length_penalty",
                    PoseT5Translator.DEFAULT_LENGTH_PENALTY,
                ),
            }
    return {
        "max_new_tokens": PoseT5Translator.DEFAULT_MAX_NEW_TOKENS,
        "beam_size": PoseT5Translator.DEFAULT_BEAM_SIZE,
        "no_repeat_ngram_size": PoseT5Translator.DEFAULT_NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": PoseT5Translator.DEFAULT_REPETITION_PENALTY,
        "length_penalty": PoseT5Translator.DEFAULT_LENGTH_PENALTY,
    }


def validate_model_dir(model_dir: str | Path) -> BundleMetadata:
    path = Path(model_dir).resolve()
    missing = [name for name in _REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise BundleValidationError(
            f"model bundle is missing required files in {path}: {', '.join(missing)}"
        )

    pose_config = _load_json(path / "pose_t5_config.json")
    feature_dim = int(pose_config.get("input_dim", 0))
    if feature_dim != 312:
        raise BundleValidationError(
            f"PoseT5 bundle at {path} must use 312-dim features; got {feature_dim}"
        )

    return BundleMetadata(
        model_dir=str(path),
        model_id=path.name,
        feature_dim=feature_dim,
        num_encoder_layers=int(pose_config["num_encoder_layers"]),
        encoder_dropout=float(pose_config["encoder_dropout"]),
        downsample_factor=int(pose_config["downsample_factor"]),
        base_model_name=str(pose_config["base_model_name"]),
        decode_config=_resolve_decode_config(path),
        missing_files=[],
    )


def resolve_model_dir_from_config(config_path: str | Path) -> Path:
    config_file = Path(config_path).resolve()
    payload = _load_json(config_file)
    preferred = payload.get("preferred_model_dirs") or []
    if not isinstance(preferred, list) or not preferred:
        raise BundleValidationError(
            f"config {config_file} must define a non-empty preferred_model_dirs list"
        )
    for candidate in preferred:
        candidate_path = (config_file.parent.parent / candidate).resolve()
        if candidate_path.is_dir():
            return candidate_path
    raise BundleValidationError(
        f"none of the configured model dirs exist for {config_file}: {preferred}"
    )


class ModelBundle:
    """Small wrapper around the existing PoseT5Translator loader."""

    def __init__(self, metadata: BundleMetadata, translator: PoseT5Translator) -> None:
        self.metadata = metadata
        self.translator = translator

    @classmethod
    def from_dir(cls, model_dir: str | Path, device: str = "cpu") -> "ModelBundle":
        metadata = validate_model_dir(model_dir)
        translator = PoseT5Translator.from_checkpoint_dir(metadata.model_dir, device=device)
        return cls(metadata=metadata, translator=translator)

    @classmethod
    def from_config(cls, config_path: str | Path, device: str = "cpu") -> "ModelBundle":
        model_dir = resolve_model_dir_from_config(config_path)
        return cls.from_dir(model_dir, device=device)

    def predict(
        self,
        features: np.ndarray,
        *,
        low_confidence_threshold: float = 0.8,
        **decode_overrides,
    ) -> dict[str, object]:
        decode_config = {**self.metadata.decode_config, **decode_overrides}
        prediction = self.translator.translate(features, **decode_config)
        return {
            "text": prediction.sentence,
            "confidence": float(prediction.score),
            "low_confidence": float(prediction.score) < low_confidence_threshold,
            "decode_config": decode_config,
            "model_id": self.metadata.model_id,
            "feature_dim": self.metadata.feature_dim,
        }

    def describe(self) -> dict[str, object]:
        return asdict(self.metadata)

