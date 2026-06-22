"""Sentence-model checkpoint discovery and metadata loading."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tsl.features.schema import infer_feature_schema_from_input_dim


_ACTIVE_MARKER_FILENAMES = ("ACTIVE", "active_checkpoint.txt")
_RUNTIME_METADATA_FILENAME = "runtime_metadata.json"
_MODEL_CONFIG_FILENAME = "model_config.json"
_MODEL_STATE_FILENAME = "slt_model.pt"
_TOKENIZER_FILENAME = "tokenizer.json"


@dataclass(frozen=True)
class SentenceModelMetadata:
    checkpoint_dir: Path
    checkpoint_name: str
    input_dim: int
    feature_schema_id: str
    tokenizer_type: str | None
    config: dict
    metadata: dict


def resolve_active_sentence_checkpoint(checkpoint_root: str | Path) -> SentenceModelMetadata:
    checkpoint_dir = _resolve_checkpoint_dir(Path(checkpoint_root))
    config = _load_json(checkpoint_dir / _MODEL_CONFIG_FILENAME)
    tokenizer = _load_json(checkpoint_dir / _TOKENIZER_FILENAME)
    runtime_metadata = _load_optional_json(checkpoint_dir / _RUNTIME_METADATA_FILENAME) or {}

    input_dim = int(config["input_dim"])
    feature_schema_id = str(
        runtime_metadata.get("feature_schema")
        or infer_feature_schema_from_input_dim(input_dim)
    )

    return SentenceModelMetadata(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=str(runtime_metadata.get("checkpoint_name") or checkpoint_dir.name),
        input_dim=input_dim,
        feature_schema_id=feature_schema_id,
        tokenizer_type=tokenizer.get("tokenizer_type"),
        config=config,
        metadata=runtime_metadata,
    )


def _resolve_checkpoint_dir(path: Path) -> Path:
    if _is_checkpoint_dir(path):
        return path

    for marker_name in _ACTIVE_MARKER_FILENAMES:
        marker_path = path / marker_name
        if marker_path.is_file():
            target_name = marker_path.read_text(encoding="utf-8").strip()
            if not target_name:
                raise FileNotFoundError(f"active checkpoint marker is empty: {marker_path!s}")
            target_dir = (path / target_name).resolve()
            if _is_checkpoint_dir(target_dir):
                return target_dir
            raise FileNotFoundError(
                f"active checkpoint {target_dir!s} from {marker_path!s} is not a valid checkpoint"
            )

    raise FileNotFoundError(
        f"no sentence checkpoint found at {path!s}; expected {_MODEL_STATE_FILENAME!r}, "
        f"{_MODEL_CONFIG_FILENAME!r}, and {_TOKENIZER_FILENAME!r}"
    )


def _is_checkpoint_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / _MODEL_STATE_FILENAME).is_file()
        and (path / _MODEL_CONFIG_FILENAME).is_file()
        and (path / _TOKENIZER_FILENAME).is_file()
    )


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"required checkpoint file not found: {path!s}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["SentenceModelMetadata", "resolve_active_sentence_checkpoint"]
