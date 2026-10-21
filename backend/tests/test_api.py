import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app

client = TestClient(app)
GOOD = {
    "age": 58, "sex": "male", "chest_pain": "asymptomatic",
    "resting_bp": 140, "cholesterol": 250,
    "fasting_blood_sugar_over_120": False, "exercise_angina": True,
}
needs_model = pytest.mark.skipif(
    not (config.MODEL_A_DIR / "model.joblib").exists(),
    reason="Model A artifacts not trained yet",
)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_rejects_bad_input():
    bad = {**GOOD, "age": 7}
    assert client.post("/predict/clinical", json=bad).status_code == 422
    bad = {**GOOD, "sex": "unknown"}
    assert client.post("/predict/clinical", json=bad).status_code == 422


@needs_model
def test_predict_shape():
    res = client.post("/predict/clinical", json=GOOD)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_band"] in {"lower", "moderate", "higher"}
    assert len(body["contributions"]) == 7
    assert body["disclaimer"]


@needs_model
def test_older_patient_scores_higher():
    young = client.post("/predict/clinical", json={**GOOD, "age": 35}).json()
    old = client.post("/predict/clinical", json={**GOOD, "age": 70}).json()
    assert old["risk_score"] >= young["risk_score"]
