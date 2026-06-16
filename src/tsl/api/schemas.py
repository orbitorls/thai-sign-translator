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
