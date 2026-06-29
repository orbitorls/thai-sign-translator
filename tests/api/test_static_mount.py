from fastapi.testclient import TestClient
from tsl.api.app import app


def test_root_serves_index_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    # App was renamed to "วาทยากร | Conductor"; assert a stable marker from the
    # served SPA shell rather than the old "Thai Sign Translator" product name.
    assert "Conductor" in resp.text
