"""FastAPI app for the Thai Sign Language translator.

Endpoints:
- POST /predict        : recognize a sequence against the registered prototypes
- POST /train-custom-sign: add a new sign (gradient-free) to the registry
- GET  /signs          : list currently registered sign names
- POST /evaluate       : run the two-track evaluation driver
- GET  /              : serve the webcam UI
- /static              : static UI assets

The default ``get_store`` / ``get_recognizer`` / ``get_eval_fn`` dependencies
load the trained encoder weights, fall back to random weights if no
checkpoint is on disk, and start from an empty prototype registry if
``prototypes.pt`` does not yet exist. They are dependency-injectable so
unit tests can substitute stubs via ``app.dependency_overrides``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from fastapi import Depends, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from tsl.api.schemas import (
    PredictRequest,
    PredictResponse,
    TrainSignRequest,
    TrainSignResponse,
)
from tsl.features.normalize import SELECTED_LANDMARKS, normalize_sequence
from tsl.inference.recognizer import Recognizer
from tsl.models.encoder import LandmarkEncoder
from tsl.registry.prototype_store import PrototypeStore

app = FastAPI(title="Thai Sign Language Translator")

_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_WEB_DIR / "index.html"))


_store: PrototypeStore | None = None
_recognizer: Recognizer | None = None


def _build_encoder() -> LandmarkEncoder:
    """Load the trained encoder weights from disk if available; else random."""
    enc = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
    weights_path = getattr(config, "ENCODER_WEIGHTS_PATH", None)
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        enc.load_state_dict(state)
    enc.eval()
    return enc


def _build_store() -> PrototypeStore:
    """Build a PrototypeStore seeded from disk, or empty if no file yet."""
    enc = _build_encoder()
    store_path = getattr(config, "PROTOTYPE_STORE_PATH", None)
    if store_path and os.path.exists(store_path):
        return PrototypeStore.load(store_path, enc)
    return PrototypeStore(enc)


def get_store() -> PrototypeStore:
    global _store
    if _store is None:
        _store = _build_store()
    return _store


def get_recognizer() -> Recognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = Recognizer(get_store())
    return _recognizer


def _raw_frames_to_array(frames) -> np.ndarray:
    return np.asarray(frames, dtype=np.float32)


def _persist_store(store) -> None:
    """Save the current prototype registry to disk.

    No-op when the store doesn't expose a save() method (test stubs) or when
    no ``PROTOTYPE_STORE_PATH`` is configured.
    """
    store_path = getattr(config, "PROTOTYPE_STORE_PATH", None)
    if not store_path or not hasattr(store, "save"):
        return
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    store.save(store_path)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, recognizer: Recognizer = Depends(get_recognizer)) -> PredictResponse:
    raw = _raw_frames_to_array(req.frames)
    seq_norm = normalize_sequence(raw)
    result = recognizer.recognize(seq_norm)
    topk = [{"word": w, "score": float(s)} for w, s in result["topk"]]
    return PredictResponse(word=result["word"], score=float(result["score"]), topk=topk)


@app.post("/train-custom-sign", response_model=TrainSignResponse)
def train_custom_sign(
    req: TrainSignRequest, store: PrototypeStore = Depends(get_store)
) -> TrainSignResponse:
    norm_clips = [normalize_sequence(_raw_frames_to_array(clip)) for clip in req.clips]
    store.add_sign(req.name, norm_clips)
    _persist_store(store)
    return TrainSignResponse(
        name=req.name,
        num_clips=len(norm_clips),
        total_signs=len(store.names()),
    )


@app.get("/signs")
def list_signs(store: PrototypeStore = Depends(get_store)) -> dict:
    return {"signs": store.names()}


def get_eval_fn():
    from tsl.eval.evaluate import run_two_track_eval

    def _run() -> dict:
        return run_two_track_eval(
            checkpoint=config.ENCODER_WEIGHTS_PATH,
            islr_parquet_dir=config.ISLR_PARQUET_DIR,
            islr_csv=config.ISLR_CSV_PATH,
            thai_root=config.THAI_DATA_DIR,
        )

    return _run


@app.post("/evaluate")
def evaluate(eval_fn=Depends(get_eval_fn)) -> dict:
    return eval_fn()
