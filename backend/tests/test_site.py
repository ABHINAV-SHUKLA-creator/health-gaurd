from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_homepage_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "HealthGuard" in res.text


def test_static_assets_served():
    assert client.get("/app.js").status_code == 200
    assert client.get("/styles.css").status_code == 200


def test_api_still_reachable():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
