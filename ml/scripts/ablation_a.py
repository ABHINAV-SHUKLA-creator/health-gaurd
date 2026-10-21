"""Ablation for Model A. Run from the project root.

Compares feature sets (full13 / no_ca_thal / checkup7) with and without
'was it missing?' flags, for logistic regression and XGBoost.
Reports pooled 5-fold CV AUC and leave-one-hospital-out AUC per hospital.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from healthguard_ml.features.preprocess import (  # noqa: E402
    FEATURE_SETS, build_preprocessor,
)


def make_model(kind, features, flags, seed):
    if kind == "logreg":
        prep = build_preprocessor(scale=True, flags=flags, features=features)
        clf = LogisticRegression(max_iter=2000)
    else:
        prep = build_preprocessor(flags=flags, features=features)
        clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed)
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_a.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    y = df[cfg["target"]]
    sites = df[cfg["group_column"]]
    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])

    rows = []
    for set_name, features in FEATURE_SETS.items():
        for flags in (True, False):
            for kind in ("logreg", "xgboost"):
                model = make_model(kind, features, flags, cfg["seed"])
                X = df[features]
                row = {
                    "features": set_name,
                    "flags": "on" if flags else "off",
                    "model": kind,
                    "cv_auc": round(cross_val_score(
                        model, X, y, cv=cv, scoring="roc_auc").mean(), 3),
                }
                for site in sorted(sites.unique()):
                    test = (sites == site).to_numpy()
                    fitted = clone(model).fit(X[~test], y[~test])
                    proba = fitted.predict_proba(X[test])[:, 1]
                    row[f"auc_{site}"] = round(roc_auc_score(y[test], proba), 3)
                site_cols = [c for c in row if c.startswith("auc_")]
                row["loso_mean"] = round(float(np.mean([row[c] for c in site_cols])), 3)
                rows.append(row)
                print(f"done {set_name} flags={row['flags']} {kind}", flush=True)

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    out = ROOT / "artifacts" / "model_a"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "ablation.csv", index=False)
    print(f"\nsaved to {out / 'ablation.csv'}")


if __name__ == "__main__":
    main()
