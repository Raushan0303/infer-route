from fastapi.testclient import TestClient
from app.main import app


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "inferroute"


def test_chat_completions_returns_valid_status():
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code in (200, 502, 503)
