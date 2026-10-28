from fastapi.testclient import TestClient

from app.core import config
from app.main import app

client = TestClient(app)


def test_site_data_endpoint():
    res = client.get("/site-data")
    if (config.ARTIFACTS / "site_data.json").exists():
        assert res.status_code == 200
        body = res.json()
        assert {"uci", "brfss", "model_a_cv", "model_a_loso"} <= set(body)
    else:
        assert res.status_code == 503
