"""Build artifacts/site_data.json and copy plots to frontend/img for the website.

Run from the project root after the data and models exist:
    python ml\\scripts\\make_site_data.py
Everything shown in the Dataset Explorer and Model Lab tabs comes from here.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
PROC = ROOT / "data" / "processed"
IMG = ROOT / "frontend" / "img"

BRFSS_AGE_LABELS = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
]
PLOTS = {
    ART / "model_a" / "v1" / "shap_summary.png": "model_a_shap.png",
    ART / "model_a" / "v1" / "reliability.png": "model_a_reliability.png",
    ART / "model_b" / "v1" / "shap_summary.png": "model_b_shap.png",
    ART / "model_b" / "v1" / "reliability.png": "model_b_reliability.png",
}


def records(path):
    if not path.exists():
        print(f"  missing {path.relative_to(ROOT)} (skipped)")
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


def group_rate(df, by):
    out = df.groupby(by, observed=True)["target"].agg(["size", "mean"]).reset_index()
    out.columns = ["group", "n", "rate"]
    return out


def uci_summary():
    path = PROC / "uci_heart.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    hospitals = group_rate(df, "site")
    ages = df.assign(band=pd.cut(df["age"], [20, 30, 40, 50, 60, 70, 80], right=False,
                                 labels=["20s", "30s", "40s", "50s", "60s", "70s"]))
    ages = group_rate(ages, "band")
    missing = df.drop(columns=["site", "target"]).isna().mean().round(3)
    return {
        "n": int(len(df)),
        "disease_rate": round(float(df["target"].mean()), 3),
        "hospitals": [
            {"hospital": r.group, "patients": int(r.n),
             "disease_rate": round(float(r.rate), 3)}
            for r in hospitals.itertuples()],
        "by_age": [{"group": str(r.group), "n": int(r.n), "rate": round(float(r.rate), 3)}
                   for r in ages.itertuples()],
        "missing": [{"feature": name, "missing_share": float(v)}
                    for name, v in missing.items() if v > 0],
    }


def brfss_summary():
    path = PROC / "brfss_heart.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["Age", "target"])
    ages = group_rate(df, "Age")
    return {
        "n": int(len(df)),
        "disease_rate": round(float(df["target"].mean()), 3),
        "by_age": [{"group": BRFSS_AGE_LABELS[int(r.group) - 1], "n": int(r.n),
                    "rate": round(float(r.rate), 3)} for r in ages.itertuples()],
    }


def main():
    print("Building site data...")
    data = {
        "uci": uci_summary(),
        "brfss": brfss_summary(),
        "model_a_cv": records(ART / "model_a" / "baseline_cv.csv"),
        "model_a_loso": records(ART / "model_a" / "baseline_loso.csv"),
        "model_a_ablation": records(ART / "model_a" / "ablation.csv"),
        "model_b_baseline": records(ART / "model_b" / "baseline.csv"),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "site_data.json").write_text(json.dumps(data, indent=1) + "\n")
    print("wrote artifacts/site_data.json")
    IMG.mkdir(parents=True, exist_ok=True)
    for src, name in PLOTS.items():
        if src.exists():
            shutil.copy(src, IMG / name)
            print(f"copied {name}")
        else:
            print(f"  missing {src.relative_to(ROOT)} (skipped)")


if __name__ == "__main__":
    main()
