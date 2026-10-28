"""Load Model A artifacts and turn a request into a score plus explanation."""
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from app.core import config
from app.schemas.clinical import ClinicalInput

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "exang"]
# Ages the training patients cover (approximately); outside it the model extrapolates.
AGE_RANGE = (28, 77)
SEX = {"female": 0, "male": 1}
CHEST_PAIN = {"typical_angina": 1, "atypical_angina": 2,
              "non_anginal": 3, "asymptomatic": 4}
LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "fbs": "Fasting blood sugar over 120 mg/dL", "exang": "Exercise-induced angina",
}


class ModelA:
    def __init__(self, directory):
        self.model = joblib.load(directory / "model.joblib")
        bundle = joblib.load(directory / "explainer.joblib")
        self.pipeline = bundle["pipeline"]
        self.explainer = bundle["explainer"]
        self.feature_names = bundle["feature_names"]
        self.metrics = json.loads((directory / "metrics.json").read_text())

    @staticmethod
    def to_frame(inp):
        row = {
            "age": inp.age,
            "sex": SEX[inp.sex],
            "cp": CHEST_PAIN[inp.chest_pain],
            "trestbps": inp.resting_bp,
            "chol": inp.cholesterol,
            "fbs": int(inp.fasting_blood_sugar_over_120),
            "exang": int(inp.exercise_angina),
        }
        return pd.DataFrame([row], columns=FEATURES).astype(float)

    @staticmethod
    def display_value(inp):
        return {
            "age": f"{inp.age} years",
            "sex": inp.sex,
            "cp": inp.chest_pain.replace("_", " "),
            "trestbps": f"{inp.resting_bp} mmHg",
            "chol": f"{inp.cholesterol} mg/dL",
            "fbs": "yes" if inp.fasting_blood_sugar_over_120 else "no",
            "exang": "yes" if inp.exercise_angina else "no",
        }

    def band(self, score):
        low, high = config.BAND_THRESHOLDS
        if score < low:
            return "lower"
        return "moderate" if score < high else "higher"

    def contributions(self, frame, inp):
        prep = self.pipeline.named_steps["prep"]
        x_t = prep.transform(frame)
        x_t = x_t.toarray() if hasattr(x_t, "toarray") else np.asarray(x_t)
        values = np.asarray(self.explainer.shap_values(x_t))[0]
        totals = dict.fromkeys(FEATURES, 0.0)
        for name, val in zip(self.feature_names, values):
            stem = name.split("__", 1)[1]
            col = stem if name.startswith("num__") else stem.rsplit("_", 1)[0]
            totals[col] += float(val)
        shown = self.display_value(inp)
        items = [
            {"feature": col, "label": LABELS[col], "value": shown[col],
             "effect": round(total, 3),
             "direction": "raises" if total >= 0 else "lowers"}
            for col, total in totals.items()
        ]
        return sorted(items, key=lambda i: abs(i["effect"]), reverse=True)

    def group_stats(self, band):
        """Observed disease rate in this band, measured on held-out hospitals."""
        table = (self.metrics.get("bands") or {}).get("table") or []
        for row in table:
            if row.get("band") == band and row.get("n"):
                return {
                    "observed_rate": float(row["observed_rate"]),
                    "n": int(row["n"]),
                    "basis": "patients in our hospital data whose answers looked like yours",
                    "outcome": "had heart disease",
                    "basis_note": (
                        "These were patients referred for heart tests, so rates are "
                        "much higher than in the general population."),
                }
        return None

    @staticmethod
    def notes(inp):
        notes = []
        low, high = AGE_RANGE
        if not low <= inp.age <= high:
            notes.append(
                f"Your age is outside the range of the patients this model learned from "
                f"(about {low} to {high}), so this result is much less reliable.")
        if inp.chest_pain == "asymptomatic":
            notes.append(
                "In this dataset, patients with no chest pain symptoms were often the ones "
                "with confirmed disease because of how patients were selected, so this "
                "answer raises the score here. It is a pattern in the data, not a medical rule.")
        return notes

    def predict(self, inp: ClinicalInput):
        frame = self.to_frame(inp)
        score = float(self.model.predict_proba(frame)[0, 1])
        band = self.band(score)
        return {
            "risk_band": band,
            "risk_score": round(score, 3),
            "contributions": self.contributions(frame, inp),
            "model_version": config.MODEL_A_VERSION,
            "disclaimer": config.DISCLAIMER,
            "notes": self.notes(inp),
            "group_stats": self.group_stats(band),
        }


@lru_cache(maxsize=1)
def get_model_a():
    return ModelA(config.MODEL_A_DIR)
