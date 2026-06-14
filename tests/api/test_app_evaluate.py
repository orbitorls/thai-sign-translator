from fastapi.testclient import TestClient
from tsl.api.app import app, get_eval_fn


def test_evaluate_returns_metrics_summary():
    def stub_eval() -> dict:
        return {"asl_held_out_acc": 0.71, "thai_acc": 0.63, "n_episodes": 100}

    app.dependency_overrides[get_eval_fn] = lambda: stub_eval
    try:
        client = TestClient(app)
        resp = client.post("/evaluate")
        assert resp.status_code == 200
        body = resp.json()
        assert body["asl_held_out_acc"] == 0.71
        assert body["thai_acc"] == 0.63
        assert body["n_episodes"] == 100
    finally:
        app.dependency_overrides.clear()
