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
