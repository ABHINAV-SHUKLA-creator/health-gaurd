"""Request and response models for the lifestyle (Model B) endpoint."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.clinical import Contribution, GroupStats


class LifestyleInput(BaseModel):
    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal["female", "male"]
    height_cm: int = Field(ge=120, le=230)
    weight_kg: float = Field(ge=30, le=300)
    smoked_100_cigarettes: bool = Field(
        description="Has smoked at least 100 cigarettes in their life")
    physically_active: bool = Field(
        description="Any physical activity in the past 30 days, outside of work")
    high_blood_pressure: bool = Field(
        description="Ever told by a health professional")
    high_cholesterol: bool = Field(
        description="Ever told by a health professional")
    diabetes: Literal["no", "prediabetes", "diabetes"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 58, "sex": "male", "height_cm": 175, "weight_kg": 90,
                "smoked_100_cigarettes": True, "physically_active": False,
                "high_blood_pressure": True, "high_cholesterol": True,
                "diabetes": "no",
            }
        }
    }

    @property
    def bmi(self):
        return round(self.weight_kg / (self.height_cm / 100) ** 2, 1)

    @model_validator(mode="after")
    def check_bmi(self):
        if not 12 <= self.bmi <= 80:
            raise ValueError(f"height and weight give an implausible BMI ({self.bmi})")
        return self


class LifestyleResult(BaseModel):
    risk_band: Literal["lower", "moderate", "higher"]
    risk_score: float = Field(description="Calibrated probability, 0-1")
    bmi: float
    contributions: list[Contribution]
    model_version: str
    disclaimer: str
    group_stats: GroupStats | None = None
