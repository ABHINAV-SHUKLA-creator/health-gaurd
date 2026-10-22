import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app
from app.services.model_b import age_category

client = TestClient(app)
GOOD = {
    "age": 58, "sex": "male", "height_cm": 175, "weight_kg": 90,
    "smoked_100_cigarettes": True, "physically_active": False,
    "high_blood_pressure": True, "high_cholesterol": True,
    "diabetes": "no",
}
needs_model = pytest.mark.skipif(
    not (config.MODEL_B_DIR / "model.joblib").exists(),
    reason="Model B artifacts not trained yet",
)


def test_age_category_boundaries():
    assert age_category(18) == 1
    assert age_category(24) == 1
    assert age_category(25) == 2
    assert age_category(29) == 2
    assert age_category(30) == 3
    assert age_category(79) == 12
    assert age_category(80) == 13
    assert age_category(100) == 13


def test_rejects_bad_input():
    assert client.post("/predict/lifestyle", json={**GOOD, "age": 5}).status_code == 422
    bad = {**GOOD, "height_cm": 120, "weight_kg": 300}
    assert client.post("/predict/lifestyle", json=bad).status_code == 422
    bad = {**GOOD, "diabetes": "maybe"}
    assert client.post("/predict/lifestyle", json=bad).status_code == 422


@needs_model
def test_predict_shape():
    res = client.post("/predict/lifestyle", json=GOOD)
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["risk_score"] <= 1
    assert body["risk_band"] in {"lower", "moderate", "higher"}
    assert body["bmi"] == 29.4
    assert len(body["contributions"]) == 8


@needs_model
def test_older_patient_scores_higher():
    young = client.post("/predict/lifestyle", json={**GOOD, "age": 30}).json()
    old = client.post("/predict/lifestyle", json={**GOOD, "age": 75}).json()
    assert old["risk_score"] >= young["risk_score"]
