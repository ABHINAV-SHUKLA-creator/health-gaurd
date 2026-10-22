"""Train, calibrate, evaluate and explain Model B. Run from project root.

Model B = logistic regression on 11 lifestyle / history features from BRFSS
(chosen from the ablation in artifacts/model_b/baseline.csv).
Evaluated on a stratified 20% holdout; final model refit on all rows.
Outputs go to artifacts/model_b/v1/.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "model_b" / "v1"

FEATURES = [
    "HighBP", "HighChol", "BMI", "Smoker", "Diabetes", "PhysActivity",
    "Sex", "Age",
]
# Age is stored as 13 five-year categories: 1 = 18-24, 2 = 25-29, ... 13 = 80+
AGE_GROUPS = {"18-39": (1, 4), "40-59": (5, 8), "60-74": (9, 11), "75+": (12, 13)}
CANDIDATE_BANDS = (0.05, 0.15)


def make_pipeline():
    return Pipeline([
        ("prep", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])


def metrics_for(y, p):
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "auc": round(roc_auc_score(y, p), 3),
        "avg_precision": round(average_precision_score(y, p), 3),
        "brier": round(brier_score_loss(y, p), 4),
    }


def subgroup_table(X, y, p):
    rows = []
    for label, mask in (
        [(f"sex={s}", (X["Sex"] == s).to_numpy()) for s in (0, 1)]
        + [(f"age {name}", X["Age"].between(lo, hi).to_numpy())
           for name, (lo, hi) in AGE_GROUPS.items()]
    ):
        if mask.sum() and y[mask].nunique() == 2:
            rows.append({"group": label, **metrics_for(y[mask], p[mask])})
    return pd.DataFrame(rows)


def band_table(y, p, cuts):
    low, high = cuts
    bands = pd.cut(p, [-0.01, low, high, 1.01],
                   labels=["lower", "moderate", "higher"])
    table = pd.DataFrame({"band": bands, "y": y.to_numpy()}).groupby(
        "band", observed=False)["y"].agg(["size", "mean"])
    table.columns = ["n", "observed_rate"]
    table["share_of_people"] = (table["n"] / table["n"].sum()).round(3)
    table["observed_rate"] = table["observed_rate"].round(3)
    return table.reset_index()


def plot_reliability(y, raw, cal, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    for label, p in (("raw", raw), ("calibrated", cal)):
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed rate")
    ax.set_title("Model B reliability (holdout)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    cfg = yaml.safe_load((ROOT / "ml" / "configs" / "model_b.yaml").read_text())
    df = pd.read_csv(ROOT / cfg["data"])
    X, y = df[FEATURES], df[cfg["target"]]
    OUT.mkdir(parents=True, exist_ok=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=cfg["seed"])
    print(f"train {len(X_tr)} rows, holdout {len(X_te)} rows, "
          f"disease rate {y.mean():.3f}")

    cv = StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
    cv_auc = cross_val_score(make_pipeline(), X_tr, y_tr, cv=cv, scoring="roc_auc")
    print(f"5-fold CV AUC on train part: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    raw_model = make_pipeline().fit(X_tr, y_tr)
    cal_model = CalibratedClassifierCV(make_pipeline(), method="sigmoid", cv=5)
    cal_model.fit(X_tr, y_tr)
    raw = raw_model.predict_proba(X_te)[:, 1]
    cal = cal_model.predict_proba(X_te)[:, 1]

    holdout = {"raw": metrics_for(y_te, raw), "calibrated": metrics_for(y_te, cal)}
    print("\nHoldout, raw:       ", holdout["raw"])
    print("Holdout, calibrated:", holdout["calibrated"])
    print(f"Brier of always predicting base rate: "
          f"{y_te.mean() * (1 - y_te.mean()):.4f}")
    plot_reliability(y_te, raw, cal, OUT / "reliability.png")

    groups = subgroup_table(X_te, y_te, pd.Series(cal, index=X_te.index))
    print("\nSubgroups (calibrated, holdout):")
    print(groups.to_string(index=False))
    bands = band_table(y_te, pd.Series(cal, index=X_te.index), CANDIDATE_BANDS)
    print(f"\nCandidate bands (cut-offs {CANDIDATE_BANDS}):")
    print(bands.to_string(index=False))

    # final models on all rows
    final_cal = CalibratedClassifierCV(make_pipeline(), method="sigmoid", cv=5)
    final_cal.fit(X, y)
    explain_pipe = make_pipeline().fit(X, y)
    sample = X.sample(min(5000, len(X)), random_state=cfg["seed"])
    X_s = explain_pipe.named_steps["prep"].transform(sample)
    masker = shap.maskers.Independent(X_s, max_samples=len(X_s))
    explainer = shap.LinearExplainer(explain_pipe.named_steps["clf"], masker)
    shap_values = explainer.shap_values(X_s)
    shap.summary_plot(shap_values, pd.DataFrame(X_s, columns=FEATURES),
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close()

    joblib.dump(final_cal, OUT / "model.joblib")
    joblib.dump({"pipeline": explain_pipe, "explainer": explainer,
                 "feature_names": FEATURES}, OUT / "explainer.joblib")
    groups.to_csv(OUT / "subgroups.csv", index=False)
    bands.to_csv(OUT / "bands.csv", index=False)
    metrics = {
        "model": "logistic_regression_calibrated_sigmoid",
        "features": FEATURES,
        "n_rows": int(len(df)),
        "disease_rate": round(float(y.mean()), 3),
        "cv_auc_mean": round(float(cv_auc.mean()), 3),
        "cv_auc_std": round(float(cv_auc.std()), 3),
        "holdout": holdout,
        "subgroups": groups.to_dict(orient="records"),
        "candidate_bands": {"cutoffs": list(CANDIDATE_BANDS),
                            "table": bands.to_dict(orient="records")},
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
