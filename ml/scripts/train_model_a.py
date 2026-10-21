"""Train, calibrate, evaluate and explain Model A. Run from project root.

Model A = logistic regression on the 7 'checkup' features, no missing flags
(chosen from the ablation in artifacts/model_a/ablation.csv).
Outputs go to artifacts/model_a/v1/.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from healthguard_ml.features.preprocess import (  # noqa: E402
    FEATURE_SETS, build_preprocessor,
)

FEATURES = FEATURE_SETS["checkup7"]
C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
OUT = ROOT / "artifacts" / "model_a" / "v1"


def make_pipeline(C):
    prep = build_preprocessor(scale=True, flags=False, features=FEATURES)
    return Pipeline([("prep", prep), ("clf", LogisticRegression(C=C, max_iter=2000))])


def loso_auc(model, X, y, sites):
    """AUC for each held-out hospital."""
    scores = {}
    for site in sorted(sites.unique()):
        test = (sites == site).to_numpy()
        fitted = clone(model).fit(X[~test], y[~test])
        scores[site] = roc_auc_score(y[test], fitted.predict_proba(X[test])[:, 1])
    return scores


def choose_C(X, y, sites):
    results = {C: np.mean(list(loso_auc(make_pipeline(C), X, y, sites).values()))
               for C in C_GRID}
    for C, score in results.items():
        print(f"  C={C:<5} leave-one-hospital-out mean AUC = {score:.3f}")
    return max(results, key=results.get)


def held_out_predictions(C, X, y, sites):
    """Raw and calibrated probabilities for each hospital, never seen in training."""
    raw = pd.Series(np.nan, index=X.index)
    cal = pd.Series(np.nan, index=X.index)
    for site in sorted(sites.unique()):
        test = (sites == site).to_numpy()
        base = make_pipeline(C)
        raw[test] = clone(base).fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
        calibrated = CalibratedClassifierCV(clone(base), method="sigmoid", cv=5)
        cal[test] = calibrated.fit(X[~test], y[~test]).predict_proba(X[test])[:, 1]
    return raw, cal


def per_site_table(y, sites, raw, cal):
    rows = []
    for site in sorted(sites.unique()):
        m = (sites == site).to_numpy()
        rows.append({
            "hospital": site,
            "n": int(m.sum()),
            "observed_rate": round(float(y[m].mean()), 3),
            "mean_pred_raw": round(float(raw[m].mean()), 3),
            "mean_pred_cal": round(float(cal[m].mean()), 3),
            "auc": round(roc_auc_score(y[m], raw[m]), 3),
            "brier_raw": round(brier_score_loss(y[m], raw[m]), 3),
            "brier_cal": round(brier_score_loss(y[m], cal[m]), 3),
        })
    return pd.DataFrame(rows)


def plot_reliability(y, raw, cal, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for label, p in (("raw", raw), ("calibrated", cal)):
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed disease rate")
    ax.set_title("Model A reliability (held-out hospitals)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_a.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    X, y, sites = df[FEATURES], df[cfg["target"]], df[cfg["group_column"]]
    OUT.mkdir(parents=True, exist_ok=True)

    print("Choosing regularisation strength C:")
    C = choose_C(X, y, sites)
    print(f"-> C = {C}\n")

    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    cv_auc = cross_val_score(make_pipeline(C), X, y, cv=cv, scoring="roc_auc")

    raw, cal = held_out_predictions(C, X, y, sites)
    table = per_site_table(y, sites, raw, cal)
    print("Held-out-hospital results:")
    print(table.to_string(index=False))
    pooled = {
        "auc": round(roc_auc_score(y, raw), 3),
        "brier_raw": round(brier_score_loss(y, raw), 3),
        "brier_cal": round(brier_score_loss(y, cal), 3),
    }
    print(f"\npooled held-out: {pooled}")
    print(f"pooled 5-fold CV AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    plot_reliability(y, raw, cal, OUT / "reliability.png")

    # final models, trained on all rows
    calibrated = CalibratedClassifierCV(make_pipeline(C), method="sigmoid", cv=5)
    calibrated.fit(X, y)
    explain_pipe = make_pipeline(C).fit(X, y)
    prep, clf = explain_pipe.named_steps["prep"], explain_pipe.named_steps["clf"]
    X_t = pd.DataFrame(to_dense(prep.transform(X)),
                       columns=prep.get_feature_names_out())
    masker = shap.maskers.Independent(X_t.to_numpy(), max_samples=len(X_t))
    explainer = shap.LinearExplainer(clf, masker)
    shap_values = explainer.shap_values(X_t.to_numpy())
    shap.summary_plot(shap_values, X_t, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    joblib.dump(calibrated, OUT / "model.joblib")
    joblib.dump({"pipeline": explain_pipe, "explainer": explainer,
                 "feature_names": list(X_t.columns)}, OUT / "explainer.joblib")
    table.to_csv(OUT / "per_hospital.csv", index=False)
    metrics = {
        "model": "logistic_regression_calibrated_sigmoid",
        "features": FEATURES,
        "C": C,
        "n_rows": int(len(df)),
        "pooled_cv_auc_mean": round(float(cv_auc.mean()), 3),
        "pooled_cv_auc_std": round(float(cv_auc.std()), 3),
        "held_out_pooled": pooled,
        "per_hospital": table.to_dict(orient="records"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
