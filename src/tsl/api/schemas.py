from pydantic import BaseModel


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
    frames: list[list[float]]
    feature_dim: int = 162
    max_len: int = 128


class TranslateSentenceResponse(BaseModel):
    sentence: str
    tokens: list[int]
    score: float
