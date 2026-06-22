from pydantic import BaseModel

from tsl.features.schema import TSL51_162


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
    frames: list  # (T, 543, 3) raw landmark frames OR (T, 312) normalized
    feature_schema: str = "raw_mediapipe_543x3"  # matches schema.py constants


class TranslateVideoResponse(BaseModel):
    sentence: str
    score: float


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
    frames: list                             # (T, 543, 3) or (T, 312)
    feature_schema: str = "raw_mediapipe_543x3"
    model: str | None = None                 # None → use default model
    max_len: int = 128


class TranslateResponse(BaseModel):
    sentence: str
    score: float
    model: str    # id of the model that was used


# --- Supported phrases schema ---

class SupportedPhrasesResponse(BaseModel):
    phrases: list[str]          # sorted list of phrases the active model recognises
    total: int                  # len(phrases)
    note: str = ""              # human-readable scope note
