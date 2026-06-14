from tsl.api.schemas import PredictRequest, PredictResponse, TrainSignRequest, TrainSignResponse


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
