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
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

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
    PredictVideoFileResponse,
    FeedbackCorrectionRequest,
    FeedbackTeachRequest,
    FeedbackSubmissionResponse,
    FeedbackStatsResponse,
    FeedbackReloadResponse,
    ConsentUpdateRequest,
    ConsentStatusResponse,
    ConsentUpdateResponse,
    DeleteDataResponse,
    FeedbackVideoResponse,
)
from tsl.api.model_catalog import (
    ModelSpec,
    get_catalog,
    get_spec,
    default_spec,
    availability,
    resolve_checkpoint_dir,
)
from tsl.features.normalize import SELECTED_LANDMARKS, normalize_sequence, normalize_sequence_v4
from tsl.inference.confidence import extract_landmark_weights, is_low_confidence
from tsl.inference.pose_t5_translator import PoseT5Prediction, PoseT5Translator
from tsl.inference.sentence_runtime import (
    FeatureSchemaMismatchError,
    SentenceRuntime,
)
from tsl.inference.video_pipeline import _extract_from_video
from tsl.inference.recognizer import Recognizer
from tsl.models.encoder import LandmarkEncoder
from tsl.models.bundle import validate_model_dir
from tsl.registry.prototype_store import PrototypeStore
from tsl.feedback.store import (
    ContributionStore,
    ContributionValidationError,
    DuplicateContributionError,
)
from tsl.feedback.rate_limit import FeedbackRateLimiter
from tsl.feedback.scheduler import start_feedback_scheduler, stop_feedback_scheduler
from tsl.privacy.consent_store import ConsentStore
from tsl.privacy.schemas import CONSENT_VERSION, ConsentScope
from tsl.privacy.user_hash import compute_user_hash
from tsl.privacy.video_store import VideoStore
from tsl.serving.cache import (
    _active_translator,
    _translator_cache,
    clear_translator_cache,
    set_active_translator_value,
)

_RAW_MEDIAPIPE_SCHEMA = "raw_mediapipe_543x3"
_RAW_MEDIAPIPE_543X4_SCHEMA = "raw_mediapipe_543x4"
_SELECTED_312_SCHEMA = "selected_312"
_MAX_VIDEO_BYTES = 25 * 1024 * 1024
_ALLOWED_VIDEO_CONTENT_TYPES = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "application/octet-stream": ".bin",
}
_MOCK_MODE_HEADER = "X-Mock-Mode"
_MOCK_MODEL_ID = "mock_v1"
_MOCK_MODEL_SPEC = ModelSpec(
    id=_MOCK_MODEL_ID,
    label_th="Conductor Demo",
    label_en="Conductor Demo",
    architecture="mock",
    checkpoint_dir="",
    bundle_config="",
    default=True,
)
_MOCK_SENTENCES = (
    "สวัสดีครับ",
    "วันนี้อากาศดี",
    "ขอบคุณมาก",
    "กำลังทดสอบระบบแปล",
    "ยินดีที่ได้รู้จัก",
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FRONTEND_DIST = _REPO_ROOT / "frontend" / "dist"
_WEB_DIR = _FRONTEND_DIST if _FRONTEND_DIST.is_dir() else _REPO_ROOT / "web"

_feedback_scheduler = None


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    global _feedback_scheduler
    _feedback_scheduler = start_feedback_scheduler()
    try:
        yield
    finally:
        stop_feedback_scheduler(_feedback_scheduler)
        _feedback_scheduler = None


app = FastAPI(title="Conductor", lifespan=_app_lifespan)

_cors_origins = [
    origin.strip()
    for origin in os.environ.get("TSL_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
_contribution_store: ContributionStore | None = None
_consent_store: ConsentStore | None = None
_feedback_rate_limiter = FeedbackRateLimiter(max_per_hour=20)


def get_contribution_store() -> ContributionStore:
    global _contribution_store
    if _contribution_store is None:
        _contribution_store = ContributionStore()
    return _contribution_store


def get_consent_store() -> ConsentStore:
    global _consent_store
    if _consent_store is None:
        _consent_store = ConsentStore()
    return _consent_store


def _truthy_header(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"true", "1", "yes"}


def _mock_mode_enabled(value: str | bool | None = None) -> bool:
    if isinstance(value, bool):
        return value
    return _truthy_header(value)


def _resolve_mock_model_id(model_id: str | None) -> str:
    if model_id is None or model_id == _MOCK_MODEL_ID:
        return _MOCK_MODEL_ID
    spec = get_spec(model_id)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"unknown model {model_id!r}")
    return spec.id


def _mock_prediction(frames, *, model_id: str):
    frame_count = len(frames) if isinstance(frames, list) else 0
    sentence = _MOCK_SENTENCES[frame_count % len(_MOCK_SENTENCES)] if frame_count else ""
    return SimpleNamespace(
        sentence=sentence,
        score=0.99 if sentence else 0.0,
        model=model_id,
    )


def _resolve_user_hash(x_user_id: str | None) -> str:
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=400, detail="X-User-Id header is required")
    try:
        return compute_user_hash(x_user_id.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_consent_scope(
    *,
    user_hash: str,
    scope: ConsentScope,
    consent_store: ConsentStore,
    header_consent: str | None = None,
) -> None:
    if not _truthy_header(header_consent) and not consent_store.has_scope(user_hash, scope):
        raise HTTPException(
            status_code=403,
            detail=f"consent required for scope: {scope}",
        )


def _require_service_consent(
    *,
    user_hash: str | None,
    consent_store: ConsentStore,
    header_consent: str | None,
) -> None:
    if not _truthy_header(header_consent):
        if user_hash is None or not consent_store.has_scope(user_hash, "service"):
            raise HTTPException(
                status_code=403,
                detail="X-Service-Consent header must be true or service consent recorded",
            )


def _require_feedback_consent(consent: str | None) -> None:
    if consent is None or consent.strip().lower() not in {"true", "1", "yes"}:
        raise HTTPException(
            status_code=403,
            detail="X-Feedback-Consent header must be true",
        )


def _feedback_session_id(request: Request, session_header: str | None) -> str:
    if session_header and session_header.strip():
        return session_header.strip()
    client = request.client
    if client and client.host:
        return client.host
    return "anonymous"


def _feedback_ip_key(request: Request) -> str:
    client = request.client
    if client and client.host:
        return f"ip:{client.host}"
    return "ip:anonymous"


def _check_feedback_rate_limit(session_id: str, ip_key: str) -> None:
    if not _feedback_rate_limiter.check_both(session_id, ip_key):
        raise HTTPException(
            status_code=429,
            detail="feedback rate limit exceeded (20 submissions per hour)",
        )


def _record_feedback_submission(session_id: str, ip_key: str) -> None:
    _feedback_rate_limiter.record_both(session_id, ip_key)


def _feedback_stats_payload(store: ContributionStore) -> dict:
    stats = store.stats()
    spec = _active_model_spec()
    stats["model"] = spec.id
    return stats


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
    checkpoint_dir = resolve_checkpoint_dir(spec)
    if spec.architecture == "pose_t5":
        from tsl.inference.pose_t5_translator import PoseT5Translator
        try:
            t = PoseT5Translator.from_checkpoint_dir(checkpoint_dir)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"model {spec.id!r} failed to load: {e}",
            ) from e
    else:
        t = SentenceRuntime.from_checkpoint_dir(checkpoint_dir)
    _translator_cache[spec.id] = t
    return t


def get_active_translator():
    """Return the best available translator for video inference.

    Preference order:
    1. Default PoseT5 catalog model when its resolved bundle is available
    2. SentenceTranslator from get_sentence_runtime() (legacy slt_v2 fallback)

    Raises HTTPException(503) if neither checkpoint is available.
    """
    cached = _active_translator
    if cached is not None:
        return cached

    spec = default_spec()
    if spec.architecture == "pose_t5" and availability(spec):
        checkpoint_dir = resolve_checkpoint_dir(spec)
        try:
            from tsl.inference.pose_t5_translator import PoseT5Translator
            translator = PoseT5Translator.from_checkpoint_dir(checkpoint_dir)
            set_active_translator_value(translator)
            return translator
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"model {spec.id!r} failed to load: {e}",
            ) from e

    runtime = get_sentence_runtime()
    set_active_translator_value(runtime)
    return runtime


def _active_model_spec():
    if _mock_mode_enabled(os.environ.get("TSL_MOCK_MODE")):
        return _MOCK_MODEL_SPEC
    spec = default_spec()
    if spec.architecture == "pose_t5" and availability(spec):
        return spec
    for candidate in get_catalog():
        if availability(candidate):
            return candidate
    return spec


@app.get("/health")
def health_check() -> dict:
    spec = _active_model_spec()
    try:
        if spec.architecture == "mock":
            return {
                "status": "ok",
                "model": spec.id,
                "architecture": spec.architecture,
                "feature_dim": 312,
                "feedback_version": get_contribution_store().feedback_version(),
                "last_retrain_at": get_contribution_store().last_retrain_at(),
            }
        if spec.architecture == "pose_t5":
            metadata = validate_model_dir(resolve_checkpoint_dir(spec))
            return {
                "status": "ok",
                "model": spec.id,
                "architecture": spec.architecture,
                "feature_dim": metadata.feature_dim,
                "feedback_version": get_contribution_store().feedback_version(),
                "last_retrain_at": get_contribution_store().last_retrain_at(),
            }
        runtime = get_sentence_runtime()
        return {
            "status": "ok",
            "model": spec.id,
            "architecture": spec.architecture,
            "feature_dim": runtime.model_metadata.input_dim,
            "feedback_version": get_contribution_store().feedback_version(),
            "last_retrain_at": get_contribution_store().last_retrain_at(),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model health check failed: {exc}") from exc


@app.get("/model-info")
def model_info() -> dict:
    spec = _active_model_spec()
    if spec.architecture == "mock":
        return {
            "model": spec.id,
            "architecture": spec.architecture,
            "feature_dim": 312,
            "generation_config": {"mode": "mock"},
            "feedback_version": get_contribution_store().feedback_version(),
            "last_retrain_at": get_contribution_store().last_retrain_at(),
        }
    if spec.architecture == "pose_t5" and availability(spec):
        metadata = validate_model_dir(resolve_checkpoint_dir(spec))
        return {
            "model": spec.id,
            "architecture": spec.architecture,
            "feature_dim": metadata.feature_dim,
            "generation_config": metadata.decode_config,
            "feedback_version": get_contribution_store().feedback_version(),
            "last_retrain_at": get_contribution_store().last_retrain_at(),
        }
    runtime = get_sentence_runtime()
    return {
        "model": spec.id,
        "architecture": spec.architecture,
        "feature_dim": runtime.model_metadata.input_dim,
        "generation_config": {"max_len": 128},
        "feedback_version": get_contribution_store().feedback_version(),
        "last_retrain_at": get_contribution_store().last_retrain_at(),
    }


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
def list_models(x_mock_mode: str | None = Header(default=None, alias=_MOCK_MODE_HEADER)) -> ModelsResponse:
    """Return all catalog models with availability and default flags."""
    if _mock_mode_enabled(x_mock_mode):
        return ModelsResponse(
            models=[
                ModelInfo(
                    id=_MOCK_MODEL_SPEC.id,
                    label_th=_MOCK_MODEL_SPEC.label_th,
                    label_en=_MOCK_MODEL_SPEC.label_en,
                    architecture=_MOCK_MODEL_SPEC.architecture,
                    available=True,
                    default=True,
                )
            ],
            default=_MOCK_MODEL_SPEC.id,
        )
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
    schema="raw_mediapipe_543x4": accepts (T, 543, 4) or flat (T, 2172).
    schema="selected_312": accepts (T, 312) directly.

    Raises HTTPException(400) on bad shape or unknown schema.
    """
    if schema == _RAW_MEDIAPIPE_543X4_SCHEMA:
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] == 4:
            features, _weights = normalize_sequence_v4(arr)
            return features
        if arr.ndim == 2 and arr.shape[1] == 543 * 4:
            features, _weights = normalize_sequence_v4(arr.reshape(arr.shape[0], 543, 4))
            return features
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] == 3:
            return normalize_sequence(arr)
        raise HTTPException(
            status_code=400,
            detail=(
                f"feature_schema={schema!r} requires shape (T, 543, 4), "
                f"(T, 2172), or legacy (T, 543, 3); got {tuple(arr.shape)}"
            ),
        )
    if schema == _RAW_MEDIAPIPE_SCHEMA:
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] in (3, 4):
            raw_xyz = arr[:, :, :3] if arr.shape[2] == 4 else arr
            return normalize_sequence(raw_xyz)
        elif arr.ndim == 2 and arr.shape[1] == 543 * 3:
            return normalize_sequence(arr.reshape(arr.shape[0], 543, 3))
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"feature_schema={schema!r} requires shape (T, 543, 3), "
                    f"(T, 543, 4), or (T, 1629); got {tuple(arr.shape)}"
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
                f"expected one of {[_RAW_MEDIAPIPE_SCHEMA, _RAW_MEDIAPIPE_543X4_SCHEMA, _SELECTED_312_SCHEMA]!r}"
            ),
        )


def _extract_landmark_weights(arr: np.ndarray, schema: str) -> np.ndarray | None:
    if schema not in (_RAW_MEDIAPIPE_SCHEMA, _RAW_MEDIAPIPE_543X4_SCHEMA):
        return None
    if schema == _RAW_MEDIAPIPE_543X4_SCHEMA:
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] == 4:
            _features, weights = normalize_sequence_v4(arr)
            return weights
        if arr.ndim == 2 and arr.shape[1] == 543 * 4:
            _features, weights = normalize_sequence_v4(arr.reshape(arr.shape[0], 543, 4))
            return weights
    if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] in (3, 4):
        return extract_landmark_weights(arr)
    if arr.ndim == 2 and arr.shape[1] == 543 * 3:
        return extract_landmark_weights(arr.reshape(arr.shape[0], 543, 3))
    if arr.ndim == 2 and arr.shape[1] == 543 * 4:
        return extract_landmark_weights(arr.reshape(arr.shape[0], 543, 4))
    return None


def _public_warning(pred) -> str | None:
    warning = getattr(pred, "warning", None)
    if warning == "low_landmark_quality":
        return "low_quality"
    if warning in ("low_light", "no_hands", "low_quality", "low_confidence"):
        return warning
    sentence = getattr(pred, "sentence", None)
    if sentence is None:
        sentence = getattr(pred, "text", "")
    if sentence and is_low_confidence(float(getattr(pred, "score", 0.0))):
        return "low_confidence"
    return None


def _translate_response_fields(pred) -> dict[str, object]:
    fields: dict[str, object] = {
        "warning": _public_warning(pred),
    }
    if isinstance(pred, PoseT5Prediction):
        fields["token_score"] = float(pred.token_score)
        fields["landmark_quality"] = float(pred.landmark_quality)
    return fields


def _translate_prediction(
    translator,
    arr: np.ndarray,
    *,
    feature_schema: str,
    max_len: int = 128,
):
    if isinstance(translator, SentenceRuntime):
        return translator.translate(
            arr,
            feature_schema=feature_schema,
            max_len=max_len,
        )
    features = _coerce_to_features(arr, feature_schema)
    weights = _extract_landmark_weights(arr, feature_schema)
    if isinstance(translator, PoseT5Translator):
        return translator.translate(features, weights=weights)
    return translator.translate(features)


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
    try:
        pred = _translate_prediction(
            translator,
            arr,
            feature_schema=req.feature_schema,
        )
    except (FeatureSchemaMismatchError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return TranslateVideoResponse(
        sentence=pred.sentence,
        score=float(pred.score),
        **_translate_response_fields(pred),
    )


@app.post("/translate", response_model=TranslateResponse, summary="Translate sign video to Thai text")
def translate(
    req: TranslateRequest,
    consent_store: ConsentStore = Depends(get_consent_store),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_service_consent: str | None = Header(default=None, alias="X-Service-Consent"),
    x_mock_mode: str | None = Header(default=None, alias=_MOCK_MODE_HEADER),
) -> TranslateResponse:
    user_hash = _resolve_user_hash(x_user_id) if x_user_id else None
    _require_service_consent(
        user_hash=user_hash,
        consent_store=consent_store,
        header_consent=x_service_consent,
    )
    return _translate_request(req, mock_mode=_mock_mode_enabled(x_mock_mode))


def _translate_request(req: TranslateRequest, *, mock_mode: bool = False) -> TranslateResponse:
    if mock_mode:
        model_id = _resolve_mock_model_id(req.model)
        pred = _mock_prediction(req.frames, model_id=model_id)
        return TranslateResponse(
            sentence=pred.sentence,
            score=float(pred.score),
            model=model_id,
        )
    if not req.frames:
        spec = get_spec(req.model) if req.model is not None else default_spec()
        if spec is None:
            raise HTTPException(status_code=400, detail=f"unknown model {req.model!r}")
        return TranslateResponse(sentence="", score=0.0, model=spec.id)

    try:
        arr = np.asarray(req.frames, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not parse frames: {e}") from e

    translator = get_translator_for(req.model)

    # Resolve the spec to get the actual model id used
    spec = get_spec(req.model) if req.model is not None else default_spec()
    used_model_id = spec.id

    try:
        pred = _translate_prediction(
            translator,
            arr,
            feature_schema=req.feature_schema,
            max_len=req.max_len,
        )
    except (FeatureSchemaMismatchError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return TranslateResponse(
        sentence=pred.sentence,
        score=float(pred.score),
        model=used_model_id,
        **_translate_response_fields(pred),
    )


@app.websocket("/ws/translate")
async def translate_websocket(websocket: WebSocket) -> None:
    """Realtime session endpoint.

    Each client message uses the same payload as POST /translate plus an
    optional request_id. Errors are returned per message so the session stays
    open while the camera keeps streaming bounded windows.
    """
    await websocket.accept()
    while True:
        try:
            payload = await websocket.receive_json()
        except WebSocketDisconnect:
            break
        except ValueError as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "request_id": None,
                    "code": 400,
                    "detail": f"invalid JSON: {e}",
                }
            )
            continue

        if not isinstance(payload, dict):
            await websocket.send_json(
                {
                    "type": "error",
                    "request_id": None,
                    "code": 400,
                    "detail": "message must be a JSON object",
                }
            )
            continue

        request_id = payload.get("request_id")
        data = dict(payload)
        data.pop("request_id", None)
        data.pop("type", None)
        mock_mode = _mock_mode_enabled(data.pop("mock_mode", None))
        service_consent = data.pop("service_consent", None)
        user_id = data.pop("user_id", None)
        started = time.perf_counter()
        try:
            if data.get("frames"):
                user_hash = compute_user_hash(str(user_id).strip()) if user_id else None
                _require_service_consent(
                    user_hash=user_hash,
                    consent_store=get_consent_store(),
                    header_consent="true" if _truthy_header(str(service_consent) if service_consent is not None else None) else None,
                )
            req = TranslateRequest(**data)
            result = _translate_request(req, mock_mode=mock_mode)
        except HTTPException as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "request_id": request_id,
                    "code": int(e.status_code),
                    "detail": str(e.detail),
                }
            )
        except (TypeError, ValueError, ValidationError) as e:
            await websocket.send_json(
                {
                    "type": "error",
                    "request_id": request_id,
                    "code": 400,
                    "detail": str(e),
                }
            )
        else:
            await websocket.send_json(
                {
                    "type": "result",
                    "request_id": request_id,
                    "sentence": result.sentence,
                    "score": result.score,
                    "model": result.model,
                    "warning": result.warning,
                    "token_score": result.token_score,
                    "landmark_quality": result.landmark_quality,
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                }
            )


@app.post("/predict/features", response_model=TranslateResponse, summary="Plan-compatible feature prediction endpoint")
def predict_features_endpoint(req: TranslateRequest) -> TranslateResponse:
    return translate(req)


@app.post("/predict/video", response_model=PredictVideoFileResponse, summary="Plan-compatible video prediction endpoint")
async def predict_video_endpoint(
    request: Request,
    translator=Depends(get_active_translator),
) -> PredictVideoFileResponse:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type not in _ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="unsupported video content-type")
    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="empty request body")
    if len(payload) > _MAX_VIDEO_BYTES:
        raise HTTPException(status_code=413, detail=f"video exceeds {_MAX_VIDEO_BYTES} bytes")

    started = time.perf_counter()
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=_ALLOWED_VIDEO_CONTENT_TYPES[content_type]) as handle:
            handle.write(payload)
            tmp_path = handle.name
        raw = _extract_from_video(tmp_path)
        if raw.shape[0] == 0:
            raise HTTPException(status_code=400, detail="video produced 0 frames")
        prediction = _translate_prediction(
            translator,
            raw,
            feature_schema=_RAW_MEDIAPIPE_SCHEMA,
        )
        if isinstance(translator, SentenceRuntime):
            feature_dim = int(translator.model_metadata.input_dim)
        else:
            feature_dim = 312
        warning = _public_warning(prediction)
        if warning is None and float(prediction.score) < 0.8 and prediction.sentence:
            warning = "low_confidence"
        return PredictVideoFileResponse(
            text=prediction.sentence,
            confidence=float(prediction.score),
            low_confidence=warning is not None,
            num_frames=int(raw.shape[0]),
            feature_dim=feature_dim,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            model=_active_model_spec().id,
            warning=warning,
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


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


@app.post("/feedback/correction", response_model=FeedbackSubmissionResponse)
def feedback_correction(
    req: FeedbackCorrectionRequest,
    request: Request,
    store: ContributionStore = Depends(get_contribution_store),
    consent_store: ConsentStore = Depends(get_consent_store),
    x_feedback_consent: str | None = Header(default=None, alias="X-Feedback-Consent"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> FeedbackSubmissionResponse:
    user_hash = _resolve_user_hash(x_user_id)
    _require_consent_scope(
        user_hash=user_hash,
        scope="model_improvement",
        consent_store=consent_store,
        header_consent=x_feedback_consent,
    )
    session_id = _feedback_session_id(request, x_session_id)
    ip_key = _feedback_ip_key(request)
    _check_feedback_rate_limit(session_id, ip_key)
    try:
        meta = store.save_correction(
            frames=req.frames,
            predicted_text=req.predicted_text,
            corrected_text=req.corrected_text,
            user_hash=user_hash,
            consent_version=CONSENT_VERSION,
            consent_scope=["model_improvement"],
            model=req.model,
            score=req.score,
            capture_quality=req.capture_quality.model_dump(exclude_none=True) if req.capture_quality else None,
        )
    except DuplicateContributionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ContributionValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _record_feedback_submission(session_id, ip_key)
    return FeedbackSubmissionResponse(
        segment_id=meta.segment_id,
        kind=meta.kind,
        status=meta.status,
        message="correction saved",
    )


@app.post("/feedback/teach", response_model=FeedbackSubmissionResponse)
def feedback_teach(
    req: FeedbackTeachRequest,
    request: Request,
    store: ContributionStore = Depends(get_contribution_store),
    consent_store: ConsentStore = Depends(get_consent_store),
    x_feedback_consent: str | None = Header(default=None, alias="X-Feedback-Consent"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> FeedbackSubmissionResponse:
    user_hash = _resolve_user_hash(x_user_id)
    _require_consent_scope(
        user_hash=user_hash,
        scope="model_improvement",
        consent_store=consent_store,
        header_consent=x_feedback_consent,
    )
    session_id = _feedback_session_id(request, x_session_id)
    ip_key = _feedback_ip_key(request)
    _check_feedback_rate_limit(session_id, ip_key)
    try:
        meta = store.save_teach(
            frames=req.frames,
            label_text=req.label_text,
            user_hash=user_hash,
            consent_version=CONSENT_VERSION,
            consent_scope=["model_improvement"],
            capture_quality=req.capture_quality.model_dump(exclude_none=True) if req.capture_quality else None,
        )
    except DuplicateContributionError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ContributionValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    _record_feedback_submission(session_id, ip_key)
    return FeedbackSubmissionResponse(
        segment_id=meta.segment_id,
        kind=meta.kind,
        status=meta.status,
        message="teach sample saved",
    )


@app.post("/feedback/video", response_model=FeedbackVideoResponse)
async def feedback_video(
    request: Request,
    store: ContributionStore = Depends(get_contribution_store),
    consent_store: ConsentStore = Depends(get_consent_store),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    x_video_consent: str | None = Header(default=None, alias="X-Video-Consent"),
) -> FeedbackVideoResponse:
    user_hash = _resolve_user_hash(x_user_id)
    _require_consent_scope(
        user_hash=user_hash,
        scope="video_research",
        consent_store=consent_store,
        header_consent=x_video_consent,
    )
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(status_code=415, detail="multipart/form-data required")
    form = await request.form()
    segment_id = str(form.get("segment_id", "")).strip()
    upload = form.get("video")
    if not segment_id:
        raise HTTPException(status_code=400, detail="segment_id is required")
    if upload is None:
        raise HTTPException(status_code=400, detail="video file is required")
    segment_meta = store.get_segment_meta(segment_id)
    if segment_meta is None:
        raise HTTPException(status_code=404, detail="segment not found")
    if segment_meta.get("user_hash") != user_hash:
        raise HTTPException(status_code=403, detail="segment does not belong to user")
    payload = await upload.read()
    if not payload:
        raise HTTPException(status_code=400, detail="empty video payload")
    video_store = VideoStore(store.root)
    try:
        video_path = video_store.save_encrypted(segment_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    store.attach_video_path(segment_id, video_path)
    return FeedbackVideoResponse(
        segment_id=segment_id,
        video_path=video_path,
        message="encrypted video saved",
    )


@app.post("/privacy/consent", response_model=ConsentUpdateResponse)
def privacy_consent_update(
    req: ConsentUpdateRequest,
    consent_store: ConsentStore = Depends(get_consent_store),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> ConsentUpdateResponse:
    user_hash = _resolve_user_hash(x_user_id)
    record = consent_store.record_consent(
        user_hash=user_hash,
        scope=req.scope,
        granted=req.granted,
        source=req.source,
        consent_version=req.consent_version,
    )
    return ConsentUpdateResponse(
        scope=record.scope,
        granted=record.granted,
        recorded_at=record.recorded_at,
    )


@app.get("/privacy/consent/status", response_model=ConsentStatusResponse)
def privacy_consent_status(
    consent_store: ConsentStore = Depends(get_consent_store),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> ConsentStatusResponse:
    user_hash = _resolve_user_hash(x_user_id)
    return ConsentStatusResponse(
        consent_version=CONSENT_VERSION,
        scopes=consent_store.current_status(user_hash),
    )


@app.post("/privacy/delete-data", response_model=DeleteDataResponse)
def privacy_delete_data(
    store: ContributionStore = Depends(get_contribution_store),
    consent_store: ConsentStore = Depends(get_consent_store),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> DeleteDataResponse:
    user_hash = _resolve_user_hash(x_user_id)
    consent_store.record_withdrawal(
        user_hash=user_hash,
        scopes=["model_improvement", "video_research", "academic_publication"],
    )
    deleted = store.delete_by_user_hash(user_hash)
    return DeleteDataResponse(
        deleted_samples=deleted,
        message="user contributions removed",
    )


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats(
    store: ContributionStore = Depends(get_contribution_store),
) -> FeedbackStatsResponse:
    payload = _feedback_stats_payload(store)
    return FeedbackStatsResponse(**payload)


@app.post("/feedback/reload-models", response_model=FeedbackReloadResponse)
def feedback_reload_models(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> FeedbackReloadResponse:
    expected = os.environ.get("TSL_FEEDBACK_ADMIN_TOKEN", "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="admin token not configured")
    if not x_admin_token or x_admin_token.strip() != expected:
        raise HTTPException(status_code=403, detail="invalid admin token")
    cleared = clear_translator_cache()
    return FeedbackReloadResponse(
        cleared_models=cleared,
        message="translator cache cleared",
    )


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> FileResponse:
    """Serve the SPA shell for client-side routes without file extensions."""
    path_name = Path(full_path).name
    if full_path.startswith(("assets/", "static/")) or "." in path_name:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(str(_WEB_DIR / "index.html"))
