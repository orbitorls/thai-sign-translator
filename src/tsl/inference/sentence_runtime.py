"""Sentence-only inference runtime with feature-schema validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from tsl.features.normalize import normalize_sequence
from tsl.features.schema import (
    RAW_MEDIAPIPE_543X3,
    RAW_MEDIAPIPE_543X4,
    SELECTED_312,
    get_feature_schema,
)
from tsl.inference.model_registry import (
    SentenceModelMetadata,
    resolve_active_sentence_checkpoint,
)
from tsl.inference.sentence_translator import SentenceTranslator


@dataclass(frozen=True)
class SentenceRuntimeResult:
    sentence: str
    score: float


class FeatureSchemaMismatchError(ValueError):
    def __init__(
        self,
        requested_schema: str,
        model_schema: str,
        message: str | None = None,
    ) -> None:
        detail = message or (
            f"feature_schema {requested_schema!r} does not match the active model "
            f"schema {model_schema!r}"
        )
        super().__init__(detail)
        self.requested_schema = requested_schema
        self.model_schema = model_schema


class SentenceRuntime:
    def __init__(self, model_metadata: SentenceModelMetadata, translator) -> None:
        self.model_metadata = model_metadata
        self.translator = translator

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_root: str,
        device: str = "cpu",
    ) -> "SentenceRuntime":
        model_metadata = resolve_active_sentence_checkpoint(checkpoint_root)
        translator = SentenceTranslator(str(model_metadata.checkpoint_dir), device=device)
        return cls(model_metadata, translator)

    def translate(
        self,
        frames,
        feature_schema: str,
        max_len: int = 128,
    ) -> SentenceRuntimeResult:
        if frames is None:
            raise ValueError("frames is required")

        if len(frames) == 0:
            return SentenceRuntimeResult(sentence="", score=0.0)

        try:
            arr = np.asarray(frames, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"could not parse frames: {exc}") from exc

        features = self._coerce_features(arr, feature_schema)
        pred = self.translator.translate(features, max_len=max_len)
        return SentenceRuntimeResult(sentence=pred.sentence, score=pred.score)

    def _coerce_features(self, arr: np.ndarray, feature_schema: str) -> np.ndarray:
        requested = get_feature_schema(feature_schema)
        model_schema = get_feature_schema(self.model_metadata.feature_schema_id)

        if requested.schema_id in (RAW_MEDIAPIPE_543X3, RAW_MEDIAPIPE_543X4):
            raw = self._validate_raw_mediapipe(arr, requested.schema_id)
            if model_schema.schema_id == SELECTED_312:
                features = normalize_sequence(raw)
            elif model_schema.schema_id == RAW_MEDIAPIPE_543X3:
                features = raw.reshape(raw.shape[0], -1)
            else:
                raise FeatureSchemaMismatchError(
                    requested_schema=requested.schema_id,
                    model_schema=model_schema.schema_id,
                    message=(
                        "raw_mediapipe_543x3 input cannot be projected to the active "
                        f"model schema {model_schema.schema_id!r}; register checkpoint "
                        "metadata or use a checkpoint trained for raw/312 features"
                    ),
                )
        elif requested.rank != 2:
            raise ValueError(
                f"feature_schema {requested.schema_id!r} expects rank {requested.rank}, "
                f"got array shape {tuple(arr.shape)}"
            )
        else:
            features = self._validate_flat_features(arr, requested.schema_id, requested.frame_dim)
            if requested.schema_id != model_schema.schema_id:
                raise FeatureSchemaMismatchError(
                    requested_schema=requested.schema_id,
                    model_schema=model_schema.schema_id,
                )

        if features.shape[1] != self.model_metadata.input_dim:
            raise FeatureSchemaMismatchError(
                requested_schema=feature_schema,
                model_schema=model_schema.schema_id,
                message=(
                    f"active model input_dim={self.model_metadata.input_dim} does not match "
                    f"prepared features with dim={features.shape[1]}"
                ),
            )
        return features.astype(np.float32, copy=False)

    def _validate_raw_mediapipe(self, arr: np.ndarray, schema_id: str) -> np.ndarray:
        coord_dim = 4 if schema_id == RAW_MEDIAPIPE_543X4 else 3
        expected_shape = (config.N_LANDMARKS, coord_dim)
        if arr.ndim != 3 or tuple(arr.shape[1:]) != expected_shape:
            raise ValueError(
                f"frames for feature_schema={schema_id!r} must have shape "
                f"(T, {config.N_LANDMARKS}, {coord_dim}); got {tuple(arr.shape)}"
            )
        return arr[:, :, :3] if coord_dim == 4 else arr

    def _validate_flat_features(
        self,
        arr: np.ndarray,
        schema_id: str,
        expected_dim: int | None,
    ) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(
                f"frames for feature_schema={schema_id!r} must be a 2-D list (T, D); "
                f"got array shape {tuple(arr.shape)}"
            )
        if expected_dim is not None and arr.shape[1] != expected_dim:
            raise ValueError(
                f"feature_schema={schema_id!r} expects D={expected_dim}; got D={arr.shape[1]}"
            )
        return arr


__all__ = [
    "FeatureSchemaMismatchError",
    "SentenceRuntime",
    "SentenceRuntimeResult",
]
