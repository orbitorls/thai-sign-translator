"""FastAPI app for the Thai Sign Language translator.

Endpoints:
- POST /predict        : legacy/dev-only word recognition against registered prototypes
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
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from tsl.api.schemas import (
    PredictRequest,
    PredictResponse,
    TrainSignRequest,
    TrainSignResponse,
    TranslateSentenceRequest,
    TranslateSentenceResponse,
    TranslateVideoRequest,
    TranslateVideoResponse,
    ModelInfo,
    ModelsResponse,
    TranslateRequest,
    TranslateResponse,
    SupportedPhrasesResponse,
)
from tsl.api.model_catalog import (
    get_catalog,
    get_spec,
    default_spec,
    availability,
)
from tsl.features.normalize import SELECTED_LANDMARKS, normalize_sequence
from tsl.inference.sentence_runtime import (
    FeatureSchemaMismatchError,
    SentenceRuntime,
)
from tsl.inference.recognizer import Recognizer
from tsl.models.encoder import LandmarkEncoder
from tsl.registry.prototype_store import PrototypeStore

_RAW_MEDIAPIPE_SCHEMA = "raw_mediapipe_543x3"
_SELECTED_312_SCHEMA = "selected_312"

app = FastAPI(title="Thai Sign Language Translator")

_REPO_ROOT = Path(__file__).resolve().parents[3]
# Production: run `cd frontend && npm run build` first so dist/ is populated.
# Vite copies index.html (with CDN <script> tags) and bundles src/ into dist/assets/.
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"
_WEB_DIR = _FRONTEND_DIST if _FRONTEND_DIST.is_dir() else _REPO_ROOT / "web"

if (_WEB_DIR / "assets").is_dir():
    # Vite build: assets are hashed under dist/assets/
    app.mount("/assets", StaticFiles(directory=str(_WEB_DIR / "assets")), name="assets")
else:
    # Legacy web/ dir: served under /static
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_WEB_DIR / "index.html"))


_store: PrototypeStore | None = None
_recognizer: Recognizer | None = None
_sentence_runtime: SentenceRuntime | None = None
_active_translator = None
_translator_cache: dict[str, object] = {}


def _build_encoder() -> LandmarkEncoder:
    """Load the trained encoder weights from disk if available; else a
    deterministic random init (fixed seed) so a saved prototype store stays
    valid across restarts."""
    weights_path = getattr(config, "ENCODER_WEIGHTS_PATH", None)
    if weights_path and os.path.exists(weights_path):
        enc = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        enc.load_state_dict(state)
    else:
        # No trained encoder shipped — seed so the random weights are
        # reproducible (prototypes.pt embeddings remain valid after restart).
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(20260629)
        try:
            enc = LandmarkEncoder(input_dim=len(SELECTED_LANDMARKS) * 3)
        finally:
            torch.random.set_rng_state(gen_state)
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


def get_sentence_runtime() -> SentenceRuntime:
    global _sentence_runtime
    if _sentence_runtime is None:
        try:
            _sentence_runtime = SentenceRuntime.from_checkpoint_dir(config.SLT_CHECKPOINT_DIR)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail="SLT checkpoint not found; train a model first",
            ) from e
    return _sentence_runtime


def get_translator_for(model_id: str | None = None):
    """Return the best available translator for the given model_id.

    model_id=None → use the default model from the catalog.
    Raises HTTPException(400) for unknown model_id.
    Raises HTTPException(503) if the model's checkpoint is not available.
    Caches loaded translators in _translator_cache.
    """
    spec = get_spec(model_id) if model_id is not None else default_spec()
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown model {model_id!r}")
    if spec.id in _translator_cache:
        return _translator_cache[spec.id]
    if not availability(spec):
        raise HTTPException(
            status_code=503,
            detail=f"model {spec.id!r} checkpoint not available",
        )
    if spec.architecture == "pose_t5":
        from tsl.inference.pose_t5_translator import PoseT5Translator
        t = PoseT5Translator.from_checkpoint_dir(spec.checkpoint_dir)
    else:
        t = SentenceRuntime.from_checkpoint_dir(spec.checkpoint_dir).translator
    _translator_cache[spec.id] = t
    return t


def get_active_translator():
    """Return the best available translator for video inference.

    Preference order:
    1. PoseT5Translator from SLT_V3_CHECKPOINT_DIR (if pose_t5_config.json exists there)
    2. SentenceTranslator from get_sentence_runtime() (legacy slt_v2 fallback)

    Raises HTTPException(503) if neither checkpoint is available.
    Backward-compat: checks and updates _active_translator global for test control.
    """
    global _active_translator
    if _active_translator is not None:
        return _active_translator

    v3_dir = getattr(config, "SLT_V3_CHECKPOINT_DIR", None)
    if v3_dir:
        v3_config_path = os.path.join(v3_dir, "pose_t5_config.json")
        if os.path.isfile(v3_config_path):
            try:
                from tsl.inference.pose_t5_translator import PoseT5Translator
                _active_translator = PoseT5Translator.from_checkpoint_dir(v3_dir)
                return _active_translator
            except Exception:
                # v3 exists but failed to load — fall through to v2
                pass

    # Fall back to legacy SentenceTranslator via SentenceRuntime
    runtime = get_sentence_runtime()  # raises 503 if not available
    _active_translator = runtime.translator
    return _active_translator


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


@app.post("/predict", response_model=PredictResponse, summary="Legacy/dev-only word prediction")
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


@app.get("/models", response_model=ModelsResponse, summary="List selectable models")
def list_models() -> ModelsResponse:
    """Return all catalog models with availability and default flags."""
    catalog = get_catalog()
    default_id = default_spec().id
    model_infos = [
        ModelInfo(
            id=spec.id,
            label_th=spec.label_th,
            label_en=spec.label_en,
            architecture=spec.architecture,
            available=availability(spec),
            default=(spec.id == default_id),
        )
        for spec in catalog
    ]
    return ModelsResponse(models=model_infos, default=default_id)


@app.post("/translate-sentence", response_model=TranslateSentenceResponse)
def translate_sentence(
    req: TranslateSentenceRequest,
    runtime: SentenceRuntime = Depends(get_sentence_runtime),
) -> TranslateSentenceResponse:
    if not req.frames:
        return TranslateSentenceResponse(sentence="", score=0.0)
    try:
        pred = runtime.translate(
            req.frames,
            feature_schema=req.feature_schema,
            max_len=req.max_len,
        )
    except (FeatureSchemaMismatchError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="SLT checkpoint not found; train a model first",
        ) from e
    return TranslateSentenceResponse(sentence=pred.sentence, score=pred.score)


def _coerce_to_features(arr: np.ndarray, schema: str) -> np.ndarray:
    """Convert raw landmark array or pre-normalized features to (T, 312) float32.

    schema="raw_mediapipe_543x3": accepts (T, 543, 3) or flat (T, 1629).
    schema="selected_312": accepts (T, 312) directly.

    Raises HTTPException(400) on bad shape or unknown schema.
    """
    if schema == _RAW_MEDIAPIPE_SCHEMA:
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] == 3:
            return normalize_sequence(arr)
        elif arr.ndim == 2 and arr.shape[1] == 543 * 3:
            return normalize_sequence(arr.reshape(arr.shape[0], 543, 3))
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"feature_schema={schema!r} requires shape (T, 543, 3) "
                    f"or (T, 1629); got {tuple(arr.shape)}"
                ),
            )
    elif schema == _SELECTED_312_SCHEMA:
        if arr.ndim != 2 or arr.shape[1] != 312:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"feature_schema={schema!r} requires shape (T, 312); "
                    f"got {tuple(arr.shape)}"
                ),
            )
        return arr
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported feature_schema={schema!r}; "
                f"expected one of {[_RAW_MEDIAPIPE_SCHEMA, _SELECTED_312_SCHEMA]!r}"
            ),
        )


@app.post("/translate-video", response_model=TranslateVideoResponse)
def translate_video_endpoint(
    req: TranslateVideoRequest,
    translator=Depends(get_active_translator),
) -> TranslateVideoResponse:
    """Translate raw landmark frames (or pre-normalized features) to Thai text.

    Accepts ``feature_schema`` values:
    - ``"raw_mediapipe_543x3"``: frames must be ``(T, 543, 3)`` or flat ``(T, 1629)``.
    - ``"selected_312"``: frames must be ``(T, 312)`` already-normalized features.

    Returns 400 for bad shape, 503 if no checkpoint is available.
    """
    if not req.frames:
        return TranslateVideoResponse(sentence="", score=0.0)
    try:
        arr = np.asarray(req.frames, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not parse frames: {e}") from e
    features = _coerce_to_features(arr, req.feature_schema)
    pred = translator.translate(features)
    return TranslateVideoResponse(sentence=pred.sentence, score=float(pred.score))


@app.post("/translate", response_model=TranslateResponse, summary="Translate sign video to Thai text")
def translate(req: TranslateRequest) -> TranslateResponse:
    """Unified translation endpoint with per-request model selection.

    Accepts raw MediaPipe frames (T, 543, 3) or pre-normalized (T, 312) features.
    Returns the Thai sentence, confidence score, and the model id used.

    model=None uses the default catalog model.
    Returns 400 for bad shape, unknown schema, or unknown model.
    Returns 503 if the selected model's checkpoint is not available.
    """
    if not req.frames:
        spec = get_spec(req.model) if req.model is not None else default_spec()
        if spec is None:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
        return TranslateResponse(sentence="", score=0.0, model=spec.id)

    try:
        arr = np.asarray(req.frames, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not parse frames: {e}") from e

    features = _coerce_to_features(arr, req.feature_schema)
    translator = get_translator_for(req.model)

    # Resolve the spec to get the actual model id used
    spec = get_spec(req.model) if req.model is not None else default_spec()
    used_model_id = spec.id

    pred = translator.translate(features)
    return TranslateResponse(
        sentence=pred.sentence,
        score=float(pred.score),
        model=used_model_id,
    )


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


@app.get("/supported-phrases", response_model=SupportedPhrasesResponse, summary="List phrases the active model recognises")
def supported_phrases() -> SupportedPhrasesResponse:
    """Return the unique sorted phrases in the TSL-51 training manifest.

    When the data directory is unavailable (e.g. CI or fresh installs) the
    endpoint returns an empty list with a scope note rather than 503, so the
    frontend can degrade gracefully.
    """
    tsl51_root = os.path.join(config.DATA_DIR, "tsl51")
    meta_path = os.path.join(tsl51_root, "metadata", "sentence_metadata.csv")

    if not os.path.isfile(meta_path):
        return SupportedPhrasesResponse(
            phrases=[],
            total=0,
            note="ข้อมูล TSL-51 ไม่พร้อมใช้งานบนเซิร์ฟเวอร์นี้",
        )

    try:
        import pandas as pd
        df = pd.read_csv(meta_path)
        col = "sentence_clean"
        if col not in df.columns:
            return SupportedPhrasesResponse(phrases=[], total=0, note="ไม่พบคอลัมน์ sentence_clean")
        phrases = sorted({
            str(v).strip()
            for v in df[col].dropna()
            if str(v).strip()
        })
        return SupportedPhrasesResponse(
            phrases=phrases,
            total=len(phrases),
            note="วลีจากชุดข้อมูล TSL-51 (ภาษามือไทย)",
        )
    except Exception as exc:  # pragma: no cover
        return SupportedPhrasesResponse(phrases=[], total=0, note=f"โหลดล้มเหลว: {exc}")
