"""Load Model B artifacts and turn a request into a score plus explanation."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.core import config
from app.schemas.lifestyle import LifestyleInput

EXPECTED = {"HighBP", "HighChol", "BMI", "Smoker", "Diabetes",
            "PhysActivity", "Sex", "Age"}
SEX = {"female": 0, "male": 1}
# Assumed coding of the Kaggle BRFSS file: 0 no diabetes, 1 prediabetes, 2 diabetes.
DIABETES = {"no": 0, "prediabetes": 1, "diabetes": 2}
LABELS = {
    "Age": "Age", "Sex": "Sex", "BMI": "Body mass index",
    "Smoker": "Smoked 100+ cigarettes", "PhysActivity": "Physically active",
    "HighBP": "High blood pressure", "HighChol": "High cholesterol",
    "Diabetes": "Diabetes status",
}


def age_category(age):
    """BRFSS 13 five-year categories: 1 = 18-24, 2 = 25-29, ... 13 = 80+."""
    if age <= 24:
        return 1
    return min((age - 25) // 5 + 2, 13)


def yes_no(flag):
    return "yes" if flag else "no"


class ModelB:
    def __init__(self, directory):
        self.model = joblib.load(directory / "model.joblib")
        bundle = joblib.load(directory / "explainer.joblib")
        self.pipeline = bundle["pipeline"]
        self.explainer = bundle["explainer"]
        self.feature_names = list(bundle["feature_names"])
        self.metrics = json.loads((directory / "metrics.json").read_text())
        if set(self.feature_names) != EXPECTED:
            raise RuntimeError(
                "Model B artifacts use a different feature set than the API "
                "expects. Re-run ml/scripts/train_model_b.py.")

    def to_frame(self, inp):
        row = {
            "HighBP": int(inp.high_blood_pressure),
            "HighChol": int(inp.high_cholesterol),
            "BMI": inp.bmi,
            "Smoker": int(inp.smoked_100_cigarettes),
            "Diabetes": DIABETES[inp.diabetes],
            "PhysActivity": int(inp.physically_active),
            "Sex": SEX[inp.sex],
            "Age": age_category(inp.age),
        }
        return pd.DataFrame([row], columns=self.feature_names).astype(float)

    @staticmethod
    def display_value(inp):
        return {
            "Age": f"{inp.age} years",
            "Sex": inp.sex,
            "BMI": f"{inp.bmi}",
            "Smoker": yes_no(inp.smoked_100_cigarettes),
            "PhysActivity": yes_no(inp.physically_active),
            "HighBP": yes_no(inp.high_blood_pressure),
            "HighChol": yes_no(inp.high_cholesterol),
            "Diabetes": inp.diabetes,
        }

    @staticmethod
    def band(score):
        low, high = config.BAND_THRESHOLDS_B
        if score < low:
            return "lower"
        return "moderate" if score < high else "higher"

    def contributions(self, frame, inp):
        x_t = self.pipeline.named_steps["prep"].transform(frame)
        values = np.asarray(self.explainer.shap_values(x_t))[0]
        shown = self.display_value(inp)
        items = [
            {"feature": name, "label": LABELS[name], "value": shown[name],
             "effect": round(float(val), 3),
             "direction": "raises" if val >= 0 else "lowers"}
            for name, val in zip(self.feature_names, values)
        ]
        return sorted(items, key=lambda i: abs(i["effect"]), reverse=True)

    def predict(self, inp: LifestyleInput):
        frame = self.to_frame(inp)
        score = float(self.model.predict_proba(frame)[0, 1])
        return {
            "risk_band": self.band(score),
            "risk_score": round(score, 3),
            "bmi": inp.bmi,
            "contributions": self.contributions(frame, inp),
            "model_version": config.MODEL_B_VERSION,
            "disclaimer": config.DISCLAIMER_B,
        }


@lru_cache(maxsize=1)
def get_model_b():
    return ModelB(config.MODEL_B_DIR)
