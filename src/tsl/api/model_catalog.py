from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import config
from tsl.inference.model_registry import _is_checkpoint_dir


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label_th: str
    label_en: str
    architecture: str          # "pose_t5" or "sentence_runtime"
    checkpoint_dir: str
    default: bool = False


_CATALOG: list[ModelSpec] = [
    ModelSpec(
        id="v3_poset5",
        label_th="PoseT5 (รุ่นล่าสุด)",
        label_en="PoseT5 (Latest)",
        architecture="pose_t5",
        checkpoint_dir=getattr(config, "SLT_V3_CHECKPOINT_DIR", ""),
        default=True,
    ),
    ModelSpec(
        id="v2_slt",
        label_th="SLT v2 (พื้นฐาน)",
        label_en="SLT v2 (Base)",
        architecture="sentence_runtime",
        checkpoint_dir=getattr(config, "SLT_CHECKPOINT_DIR", ""),
    ),
    ModelSpec(
        id="combined",
        label_th="รวมชุดข้อมูล",
        label_en="Combined Dataset",
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


def availability(spec: ModelSpec) -> bool:
    """Return True if the model's checkpoint is present and loadable."""
    d = spec.checkpoint_dir
    if not d:
        return False
    if spec.architecture == "pose_t5":
        return os.path.isfile(os.path.join(d, "pose_t5_config.json"))
    else:  # sentence_runtime
        return _is_checkpoint_dir(Path(d))
