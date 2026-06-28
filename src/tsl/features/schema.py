"""Canonical feature schema identifiers for inference and training."""
from __future__ import annotations

from dataclasses import dataclass


RAW_MEDIAPIPE_543X3 = "raw_mediapipe_543x3"
RAW_MEDIAPIPE_543X4 = "raw_mediapipe_543x4"
SELECTED_312 = "selected_312"
TSL51_162 = "tsl51_162"


@dataclass(frozen=True)
class FeatureSchema:
    schema_id: str
    rank: int
    frame_dim: int | None
    description: str
    landmark_count: int | None = None
    coord_dim: int | None = None


_SCHEMAS: dict[str, FeatureSchema] = {
    RAW_MEDIAPIPE_543X3: FeatureSchema(
        schema_id=RAW_MEDIAPIPE_543X3,
        rank=3,
        frame_dim=None,
        description="Raw MediaPipe Holistic landmarks with shape (T, 543, 3).",
        landmark_count=543,
        coord_dim=3,
    ),
    RAW_MEDIAPIPE_543X4: FeatureSchema(
        schema_id=RAW_MEDIAPIPE_543X4,
        rank=3,
        frame_dim=None,
        description=(
            "Raw MediaPipe Holistic landmarks with shape (T, 543, 4); "
            "the fourth channel is visibility/presence weight."
        ),
        landmark_count=543,
        coord_dim=4,
    ),
    SELECTED_312: FeatureSchema(
        schema_id=SELECTED_312,
        rank=2,
        frame_dim=312,
        description="Normalized 104-landmark subset flattened to 312 floats per frame.",
    ),
    TSL51_162: FeatureSchema(
        schema_id=TSL51_162,
        rank=2,
        frame_dim=162,
        description="TSL-51 flattened landmark layout with 162 floats per frame.",
    ),
}


def get_feature_schema(schema_id: str) -> FeatureSchema:
    try:
        return _SCHEMAS[schema_id]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature_schema={schema_id!r}; expected one of {sorted(_SCHEMAS)}"
        ) from exc


def infer_feature_schema_from_input_dim(input_dim: int) -> str:
    if input_dim == 312:
        return SELECTED_312
    if input_dim == 162:
        return TSL51_162
    raise ValueError(
        "could not infer feature schema from input_dim="
        f"{input_dim}; add runtime metadata for this checkpoint"
    )


__all__ = [
    "FeatureSchema",
    "RAW_MEDIAPIPE_543X3",
    "RAW_MEDIAPIPE_543X4",
    "SELECTED_312",
    "TSL51_162",
    "get_feature_schema",
    "infer_feature_schema_from_input_dim",
]
