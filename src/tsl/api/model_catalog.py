from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import config
from tsl.inference.model_registry import _is_checkpoint_dir
from tsl.models.bundle import BundleValidationError, resolve_model_dir_from_config, validate_model_dir


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label_th: str
    label_en: str
    architecture: str          # "pose_t5" or "sentence_runtime"
    checkpoint_dir: str
    bundle_config: str = ""
    default: bool = False


_CATALOG: list[ModelSpec] = [
    ModelSpec(
        id="v3_poset5",
        label_th="Conductor Core",
        label_en="Conductor Core",
        architecture="pose_t5",
        checkpoint_dir=getattr(config, "SLT_V3_CHECKPOINT_DIR", ""),
        bundle_config=getattr(config, "SLT_V3_MODEL_CONFIG", ""),
        default=True,
    ),
    ModelSpec(
        id="v2_slt",
        label_th="Conductor Base",
        label_en="Conductor Base",
        architecture="sentence_runtime",
        checkpoint_dir=getattr(config, "SLT_CHECKPOINT_DIR", ""),
    ),
    ModelSpec(
        id="combined",
        label_th="Conductor Fusion",
        label_en="Conductor Fusion",
        architecture="sentence_runtime",
        checkpoint_dir=getattr(config, "SLT_COMBINED_CHECKPOINT_DIR", ""),
    ),
]


def get_catalog() -> list[ModelSpec]:
    return _CATALOG


def get_spec(model_id: str) -> ModelSpec | None:
    for spec in _CATALOG:
        if spec.id == model_id:
            return spec
    return None


def default_spec() -> ModelSpec:
    for spec in _CATALOG:
        if spec.default:
            return spec
    return _CATALOG[0]


def resolve_checkpoint_dir(spec: ModelSpec) -> str:
    """Resolve the runtime directory that should actually be loaded."""
    if spec.architecture == "pose_t5" and spec.bundle_config:
        return str(resolve_model_dir_from_config(spec.bundle_config))
    if not spec.checkpoint_dir:
        raise FileNotFoundError(f"model {spec.id!r} does not declare a checkpoint directory")
    return spec.checkpoint_dir


def availability(spec: ModelSpec) -> bool:
    """Return True if the model's checkpoint is present and loadable."""
    try:
        d = resolve_checkpoint_dir(spec)
        if spec.architecture == "pose_t5":
            validate_model_dir(d)
            return True
        return _is_checkpoint_dir(Path(d))
    except (BundleValidationError, FileNotFoundError, OSError, ValueError):
        return False
