import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app

client = TestClient(app)
BASE = {
    "age": 50, "sex": "male", "chest_pain": "non_anginal",
    "resting_bp": 130, "cholesterol": 220,
    "fasting_blood_sugar_over_120": False, "exercise_angina": False,
}
needs_model = pytest.mark.skipif(
    not (config.MODEL_A_DIR / "model.joblib").exists(),
    reason="Model A artifacts not trained yet",
)


@needs_model
def test_no_notes_for_typical_case():
    assert client.post("/predict/clinical", json=BASE).json()["notes"] == []


@needs_model
def test_note_for_age_outside_training_range():
    body = client.post("/predict/clinical", json={**BASE, "age": 22}).json()
    assert any("outside the range" in n for n in body["notes"])


@needs_model
def test_note_for_asymptomatic():
    body = client.post(
        "/predict/clinical", json={**BASE, "chest_pain": "asymptomatic"}).json()
    assert any("no chest pain symptoms" in n for n in body["notes"])
