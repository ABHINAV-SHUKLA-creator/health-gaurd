"""Baseline + feature-set ablation for Model B (BRFSS). Run from project root.

Heart disease is rare in this data (~1 in 11), so accuracy is meaningless.
We report ROC AUC, average precision (PR AUC) and Brier score, all from
5-fold stratified cross-validation.
"""
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]

ALL = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income",
]
# Variables that can be *consequences* of heart disease (the target is
# "has had heart disease/attack"), so using them partly predicts the outcome
# from its own effects.
CONSEQUENCE_LIKE = ["Stroke", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk"]
# Healthcare access / socioeconomic columns: weak health signal, and poor
# inputs for a public screening form.
ACCESS_SES = ["CholCheck", "AnyHealthcare", "NoDocbcCost", "Education", "Income"]

FEATURE_SETS = {
    "all21": ALL,
    "no_consequence16": [c for c in ALL if c not in CONSEQUENCE_LIKE],
    "lifestyle11": [c for c in ALL
                    if c not in CONSEQUENCE_LIKE and c not in ACCESS_SES],
}


def make_model(kind, seed):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
        random_state=seed, n_jobs=-1)


def fmt(scores):
    return f"{scores.mean():.3f} +/- {scores.std():.3f}"


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_b.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    y = df[cfg["target"]]
    rate = float(y.mean())
    print(f"rows: {len(df)}  disease rate: {rate:.3f}")
    print(f"Brier score of always predicting the base rate: {rate * (1 - rate):.3f}\n")

    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    scoring = {"auc": "roc_auc", "ap": "average_precision",
               "brier": "neg_brier_score"}
    rows = []
    for set_name, features in FEATURE_SETS.items():
        for kind in ("logreg", "xgboost"):
            start = time.time()
            res = cross_validate(make_model(kind, cfg["seed"]), df[features], y,
                                 cv=cv, scoring=scoring)
            rows.append({
                "features": set_name,
                "n_features": len(features),
                "model": kind,
                "auc": fmt(res["test_auc"]),
                "avg_precision": fmt(res["test_ap"]),
                "brier": f"{-res['test_brier'].mean():.4f}",
            })
            print(f"done {set_name} {kind} ({time.time() - start:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    out = ROOT / "artifacts" / "model_b"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "baseline.csv", index=False)
    print(f"\nsaved to {out / 'baseline.csv'}")


if __name__ == "__main__":
    main()
