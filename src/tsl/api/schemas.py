from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from tsl.features.schema import RAW_MEDIAPIPE_543X3, RAW_MEDIAPIPE_543X4, TSL51_162
from tsl.feedback.store import MAX_FRAMES, MIN_FRAMES


class PredictRequest(BaseModel):
    frames: list[list[list[float]]]


class PredictResponse(BaseModel):
    word: str
    score: float
    topk: list[dict]


class TrainSignRequest(BaseModel):
    name: str
    clips: list[list[list[list[float]]]]


class TrainSignResponse(BaseModel):
    name: str
    num_clips: int
    total_signs: int


class TranslateSentenceRequest(BaseModel):
    frames: list
    feature_schema: str = TSL51_162
    max_len: int = 128


class TranslateSentenceResponse(BaseModel):
    sentence: str
    score: float


class TranslateVideoRequest(BaseModel):
    frames: list  # (T, 543, 3|4) raw landmark frames OR (T, 312) normalized
    feature_schema: str = RAW_MEDIAPIPE_543X3


class TranslateVideoResponse(BaseModel):
    sentence: str
    score: float
    warning: str | None = None
    token_score: float | None = None
    landmark_quality: float | None = None


# --- Model catalog schemas ---

class ModelInfo(BaseModel):
    id: str
    label_th: str
    label_en: str
    architecture: str   # "pose_t5" or "sentence_runtime"
    available: bool
    default: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default: str   # id of the default model


# --- Unified translate schema ---

class TranslateRequest(BaseModel):
    frames: list                             # (T, 543, 3|4) or (T, 312)
    feature_schema: str = RAW_MEDIAPIPE_543X3
    model: str | None = None                 # None → use default model
    max_len: int = 128


class TranslateResponse(BaseModel):
    sentence: str
    score: float
    model: str    # id of the model that was used
    warning: str | None = None
    token_score: float | None = None
    landmark_quality: float | None = None


# --- Supported phrases schema ---

class SupportedPhrasesResponse(BaseModel):
    phrases: list[str]          # sorted list of phrases the active model recognises
    total: int                  # len(phrases)
    note: str = ""              # human-readable scope note


class PredictVideoFileResponse(BaseModel):
    text: str
    confidence: float
    low_confidence: bool
    num_frames: int
    feature_dim: int
    latency_ms: float
    model: str
    warning: str | None = None


# --- User feedback schemas ---

class FeedbackFramesMixin(BaseModel):
    frames: list

    @model_validator(mode="after")
    def validate_frames_shape(self) -> "FeedbackFramesMixin":
        frames = self.frames
        if not isinstance(frames, list):
            raise ValueError("frames must be a list")
        if len(frames) < MIN_FRAMES:
            raise ValueError(f"at least {MIN_FRAMES} frames required")
        if len(frames) > MAX_FRAMES:
            raise ValueError(f"at most {MAX_FRAMES} frames allowed")
        for frame in frames:
            if not isinstance(frame, list) or len(frame) != 543:
                raise ValueError("each frame must have 543 landmarks")
            for landmark in frame:
                if not isinstance(landmark, list) or len(landmark) not in (3, 4):
                    raise ValueError("each landmark must have 3 or 4 coordinates")
        return self


class CaptureQualityPayload(BaseModel):
    fps: float | None = None
    lighting_ok: bool | None = None
    hand_present: bool | None = None
    warning: Literal[
        "low_light",
        "no_hands",
        "low_fps",
        "motion_blur",
        "low_quality",
        "low_confidence",
    ] | None = None
    landmark_quality: float | None = None
    feature_schema: Literal[RAW_MEDIAPIPE_543X3, RAW_MEDIAPIPE_543X4] | None = None
    camera_facing: Literal["user", "environment"] | None = None

    @field_validator("fps")
    @classmethod
    def clamp_fps(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return min(max(float(value), 0.0), 120.0)

    @field_validator("landmark_quality")
    @classmethod
    def clamp_landmark_quality(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return min(max(float(value), 0.0), 1.0)


class FeedbackCorrectionRequest(FeedbackFramesMixin):
    predicted_text: str
    corrected_text: str
    model: str | None = None
    score: float | None = None
    capture_quality: CaptureQualityPayload | None = None


class FeedbackTeachRequest(FeedbackFramesMixin):
    label_text: str
    capture_quality: CaptureQualityPayload | None = None


class FeedbackSubmissionResponse(BaseModel):
    segment_id: str
    kind: str
    status: str
    message: str


class FeedbackStatsResponse(BaseModel):
    pending_count: int
    total_count: int
    last_retrain_at: str | None = None
    last_attempt_at: str | None = None
    feedback_version: str | None = None
    model: str | None = None


class FeedbackReloadResponse(BaseModel):
    cleared_models: int
    message: str


# --- Privacy / consent schemas ---

class ConsentUpdateRequest(BaseModel):
    scope: Literal[
        "service",
        "model_improvement",
        "video_research",
        "academic_publication",
    ]
    granted: bool
    consent_version: str = "2026-06-28-v1"
    source: Literal["consent_modal", "settings_toggle", "api", "withdrawal"] = "api"


class ConsentStatusResponse(BaseModel):
    consent_version: str
    scopes: dict[str, bool]


class ConsentUpdateResponse(BaseModel):
    scope: str
    granted: bool
    recorded_at: str


class DeleteDataResponse(BaseModel):
    deleted_samples: int
    message: str


class FeedbackVideoResponse(BaseModel):
    segment_id: str
    video_path: str
    message: str
