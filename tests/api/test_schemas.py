from tsl.api.schemas import (
    PredictRequest,
    PredictResponse,
    TrainSignRequest,
    TrainSignResponse,
    TranslateSentenceRequest,
    TranslateSentenceResponse,
)
from tsl.features.schema import TSL51_162


def test_predict_request_holds_raw_frames():
    frame = [[0.0, 0.0, 0.0] for _ in range(543)]
    req = PredictRequest(frames=[frame, frame])
    assert len(req.frames) == 2
    assert len(req.frames[0]) == 543
    assert len(req.frames[0][0]) == 3


def test_predict_response_shape():
    resp = PredictResponse(
        word="hello",
        score=0.91,
        topk=[{"word": "hello", "score": 0.91}, {"word": "bye", "score": 0.42}],
    )
    assert resp.word == "hello"
    assert resp.score == 0.91
    assert resp.topk[0]["word"] == "hello"
    dumped = resp.model_dump()
    assert dumped["topk"][1]["score"] == 0.42


def test_train_sign_request_holds_clips_of_frames():
    frame = [[0.0, 0.0, 0.0] for _ in range(543)]
    clip = [frame, frame, frame]
    req = TrainSignRequest(name="cat", clips=[clip, clip])
    assert req.name == "cat"
    assert len(req.clips) == 2
    assert len(req.clips[0]) == 3
    assert len(req.clips[0][0]) == 543


def test_train_sign_response_shape():
    resp = TrainSignResponse(name="cat", num_clips=2, total_signs=5)
    assert resp.name == "cat"
    assert resp.num_clips == 2
    assert resp.total_signs == 5


def test_translate_sentence_request_holds_schema_and_frames():
    frame = [0.0] * 162
    req = TranslateSentenceRequest(frames=[frame, frame], feature_schema=TSL51_162)
    assert req.feature_schema == TSL51_162
    assert len(req.frames) == 2
    assert len(req.frames[0]) == 162


def test_translate_sentence_response_shape():
    resp = TranslateSentenceResponse(sentence="hello", score=0.91)
    assert resp.sentence == "hello"
    assert resp.score == 0.91


# --- New schemas from Task 2 ---
from tsl.api.schemas import ModelInfo, ModelsResponse, TranslateRequest, TranslateResponse


def test_model_info_schema():
    m = ModelInfo(
        id="v3_poset5",
        label_th="PoseT5 (รุ่นล่าสุด)",
        label_en="PoseT5 (Latest)",
        architecture="pose_t5",
        available=True,
        default=True,
    )
    assert m.id == "v3_poset5"
    assert m.available is True
    assert m.default is True


def test_models_response_schema():
    m = ModelInfo(id="v2_slt", label_th="SLT", label_en="SLT", architecture="sentence_runtime", available=False, default=False)
    resp = ModelsResponse(models=[m], default="v2_slt")
    assert resp.default == "v2_slt"
    assert len(resp.models) == 1


def test_translate_request_defaults():
    req = TranslateRequest(frames=[[0.0] * 312])
    assert req.feature_schema == "raw_mediapipe_543x3"
    assert req.model is None
    assert req.max_len == 128


def test_translate_request_with_model():
    req = TranslateRequest(frames=[], model="v2_slt")
    assert req.model == "v2_slt"


def test_translate_response_includes_model_field():
    resp = TranslateResponse(sentence="สวัสดี", score=0.9, model="v3_poset5")
    assert resp.sentence == "สวัสดี"
    assert resp.model == "v3_poset5"
