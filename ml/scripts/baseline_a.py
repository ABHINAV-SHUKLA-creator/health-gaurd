"""Compare baseline models for Model A. Run from the project root.

Two tests:
  1. 5-fold stratified cross-validation on all hospitals pooled
  2. leave-one-hospital-out: train on 3 hospitals, test on the 4th
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from healthguard_ml.features.preprocess import (  # noqa: E402
    FEATURES, build_preprocessor,
)


def make_models(seed):
    return {
        "logreg": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, random_state=seed)),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=seed)),
        ]),
    }


def cross_val_table(models, X, y, folds, seed):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"auc": "roc_auc", "acc": "accuracy",
               "brier": "neg_brier_score"}
    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "auc": f"{res['test_auc'].mean():.3f} +/- {res['test_auc'].std():.3f}",
            "accuracy": f"{res['test_acc'].mean():.3f} +/- {res['test_acc'].std():.3f}",
            "brier": f"{-res['test_brier'].mean():.3f}",
        })
    return pd.DataFrame(rows)


def leave_one_site_out(models, X, y, sites):
    rows = []
    for name, model in models.items():
        for site in sorted(sites.unique()):
            test = (sites == site).to_numpy()
            fitted = clone(model).fit(X[~test], y[~test])
            proba = fitted.predict_proba(X[test])[:, 1]
            both = y[test].nunique() == 2
            rows.append({
                "model": name,
                "held_out": site,
                "n_test": int(test.sum()),
                "disease_rate": round(float(y[test].mean()), 3),
                "auc": round(roc_auc_score(y[test], proba), 3) if both else np.nan,
                "accuracy": round(accuracy_score(y[test], proba > 0.5), 3),
                "brier": round(brier_score_loss(y[test], proba), 3),
            })
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_a.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    X, y = df[FEATURES], df[cfg["target"]]
    sites = df[cfg["group_column"]]
    models = make_models(cfg["seed"])

    print(f"rows: {len(df)}  disease rate: {y.mean():.3f}\n")
    print("== 5-fold CV (all hospitals pooled)")
    cv_table = cross_val_table(models, X, y, cfg["cv_folds"], cfg["seed"])
    print(cv_table.to_string(index=False))

    print("\n== Leave-one-hospital-out")
    loso = leave_one_site_out(models, X, y, sites)
    print(loso.to_string(index=False))

    out = ROOT / "artifacts" / "model_a"
    out.mkdir(parents=True, exist_ok=True)
    cv_table.to_csv(out / "baseline_cv.csv", index=False)
    loso.to_csv(out / "baseline_loso.csv", index=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
