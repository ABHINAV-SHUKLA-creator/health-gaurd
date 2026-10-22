"""Does dropping misleading / weak features from Model B cost anything?

Run from project root. Logistic regression, 5-fold stratified CV, same seed
as the other Model B scripts.
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

CURRENT = [
    "HighBP", "HighChol", "BMI", "Smoker", "Diabetes", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump", "Sex", "Age",
]
SETS = {
    "current11": CURRENT,
    "no_alcohol10": [f for f in CURRENT if f != "HvyAlcoholConsump"],
    "no_alcohol_diet8": [f for f in CURRENT
                         if f not in ("HvyAlcoholConsump", "Fruits", "Veggies")],
}


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_b.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    y = df[cfg["target"]]
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    scoring = {"auc": "roc_auc", "ap": "average_precision",
               "brier": "neg_brier_score"}
    rows = []
    for name, features in SETS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        res = cross_validate(model, df[features], y, cv=cv, scoring=scoring)
        rows.append({
            "features": name,
            "n": len(features),
            "auc": f"{res['test_auc'].mean():.4f} +/- {res['test_auc'].std():.4f}",
            "avg_precision": f"{res['test_ap'].mean():.4f}",
            "brier": f"{-res['test_brier'].mean():.4f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
