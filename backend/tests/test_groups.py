import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import app

client = TestClient(app)
CLINICAL = {
    "age": 50, "sex": "male", "chest_pain": "non_anginal",
    "resting_bp": 130, "cholesterol": 220,
    "fasting_blood_sugar_over_120": False, "exercise_angina": False,
}
LIFESTYLE = {
    "age": 58, "sex": "male", "height_cm": 175, "weight_kg": 90,
    "smoked_100_cigarettes": True, "physically_active": False,
    "high_blood_pressure": True, "high_cholesterol": True, "diabetes": "no",
}
needs_models = pytest.mark.skipif(
    not (config.MODEL_A_DIR / "model.joblib").exists()
    or not (config.MODEL_B_DIR / "model.joblib").exists(),
    reason="models not trained yet",
)


def check(stats):
    if stats is not None:  # present once the training scripts have been re-run
        assert 0 <= stats["observed_rate"] <= 1
        assert stats["n"] > 0
        assert stats["basis"] and stats["outcome"] and stats["basis_note"]


@needs_models
def test_clinical_group_stats():
    body = client.post("/predict/clinical", json=CLINICAL).json()
    assert "group_stats" in body
    check(body["group_stats"])


@needs_models
def test_lifestyle_group_stats():
    body = client.post("/predict/lifestyle", json=LIFESTYLE).json()
    assert "group_stats" in body
    check(body["group_stats"])
