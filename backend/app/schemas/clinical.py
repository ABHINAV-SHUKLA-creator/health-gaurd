"""Request and response models for the clinical (Model A) endpoint."""
from typing import Literal

from pydantic import BaseModel, Field


class ClinicalInput(BaseModel):
    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal["female", "male"]
    chest_pain: Literal[
        "typical_angina", "atypical_angina", "non_anginal", "asymptomatic"
    ]
    resting_bp: int = Field(ge=80, le=250, description="Resting blood pressure, mmHg")
    cholesterol: int = Field(ge=100, le=600, description="Serum cholesterol, mg/dL")
    fasting_blood_sugar_over_120: bool
    exercise_angina: bool = Field(description="Chest pain brought on by exercise")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58, "sex": "male", "chest_pain": "asymptomatic",
                "resting_bp": 140, "cholesterol": 250,
                "fasting_blood_sugar_over_120": False, "exercise_angina": True,
            }
        }
    }


class Contribution(BaseModel):
    feature: str
    label: str
    value: str
    effect: float = Field(description="Push on the risk score, in log-odds units")
    direction: Literal["raises", "lowers"]


class ClinicalResult(BaseModel):
    risk_band: Literal["lower", "moderate", "higher"]
    risk_score: float = Field(description="Calibrated score 0-1, relative to training data")
    contributions: list[Contribution]
    model_version: str
    disclaimer: str
