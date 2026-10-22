"""Write docs/model_cards/*.md from the saved metrics.json files.

Run from the project root after training:  python ml\\scripts\\make_model_cards.py
Numbers come straight from the artifacts, so the cards never go stale.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "docs" / "model_cards"


def md_table(rows, columns):
    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r.get(key, "-")) for key, _ in columns) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def card_a(m):
    hospitals = md_table(m.get("per_hospital", []), [
        ("hospital", "Held-out hospital"), ("n", "Patients"),
        ("observed_rate", "Actual disease rate"),
        ("mean_pred_cal", "Mean predicted score"), ("auc", "AUC"),
        ("brier_cal", "Brier"),
    ])
    pooled = m.get("held_out_pooled", {})
    return f"""# Model card: Model A (Clinical check)

## Intended use
Educational demonstration of explainable risk estimation from basic checkup values.
Not a medical device. Not for diagnosis, treatment or screening decisions.

## Data
UCI Heart Disease, four sites (Cleveland, Hungarian, Switzerland, VA Long Beach),
{m.get("n_rows")} patients (CC BY 4.0). Target: any angiographic disease (original
label greater than 0). Zero values for cholesterol and blood pressure were treated as
missing and filled with the median.

## Features ({len(m.get("features", []))})
{", ".join(m.get("features", []))}

## Model
Standardised logistic regression (C = {m.get("C")}), sigmoid-calibrated, with exact SHAP
(linear explainer) for per-person explanations. Chosen over random forest and XGBoost
because it matched them on held-out hospitals and is simpler to explain.

## Performance
- Pooled 5-fold cross-validation AUC: {m.get("pooled_cv_auc_mean")}
  (std {m.get("pooled_cv_auc_std")})
- Pooled held-out-hospital AUC: {pooled.get("auc")}; Brier {pooled.get("brier_cal")}
  (always predicting the base rate would score about 0.247)

Leave-one-hospital-out (train on three hospitals, test on the fourth):

{hospitals}

## Known limitations
- Small (under 1,000 patients), from the 1980s, from referred patients at four hospitals.
- Disease rates differ widely by hospital, so predicted scores are *relative to the
  training mix*, not absolute probabilities. Calibration cannot remove this.
- Switzerland has very few patients without disease, so its numbers are unstable.
- "No chest pain symptoms" raises the score because of how patients were selected;
  this is a pattern in the data, not a medical rule.
- Not tested on any population outside these hospitals. Ages far outside about 28-77
  are extrapolation.
- The risk-band cut-offs are provisional.

Generated at: {m.get("trained_at")}
"""


def card_b(m):
    hold = m.get("holdout", {}).get("calibrated", {})
    groups = md_table(m.get("subgroups", []), [
        ("group", "Group"), ("n", "People"), ("positives", "Reported disease"),
        ("auc", "AUC"), ("avg_precision", "Avg. precision"), ("brier", "Brier"),
    ])
    bands = md_table(m.get("candidate_bands", {}).get("table", []), [
        ("band", "Result group"), ("n", "People"),
        ("observed_rate", "Reported disease"), ("share_of_people", "Share of people"),
    ])
    cutoffs = m.get("candidate_bands", {}).get("cutoffs")
    return f"""# Model card: Model B (Lifestyle check)

## Intended use
Educational demonstration of an explainable screening-style estimate from answers a
person can give without any tests. Not a medical device. Not for medical decisions.

## Data
CDC BRFSS 2015 survey, cleaned version distributed on Kaggle. {m.get("n_rows")} rows after
removing duplicates. Target: respondent *reported ever having* coronary heart disease or a
heart attack ({round(100 * (m.get("disease_rate") or 0), 1)}% of rows). Survey sampling weights
are not included, so rates describe this file, not the US population.

## Features ({len(m.get("features", []))})
{", ".join(m.get("features", []))}

Columns deliberately left out:
- Stroke, general/physical/mental health days, difficulty walking: these can be
  consequences of heart disease, so they partly predict the outcome from its own effects.
- Income, education, insurance and healthcare-access columns: weak health signal and poor
  inputs for a public screening form.
- Heavy alcohol use: showed a protective sign that is likely a survey artefact, not a
  health effect (not verified); dropping it cost no measurable accuracy.
- Fruit and vegetable intake: no measurable signal in this data (this does not mean diet
  does not matter).

## Model
Standardised logistic regression, sigmoid-calibrated, with exact SHAP (linear explainer).

## Performance
- 5-fold cross-validation AUC: {m.get("cv_auc_mean")} (std {m.get("cv_auc_std")})
- Stratified 20% holdout (calibrated): AUC {hold.get("auc")}, average precision
  {hold.get("avg_precision")}, Brier {hold.get("brier")}

By group (holdout):

{groups}

Result groups (cut-offs {cutoffs}), observed on the holdout:

{bands}

## Known limitations
- Self-reported, cross-sectional data: the model describes who *reports* heart disease,
  it does not predict future disease.
- US adults in 2015; not validated elsewhere. Age is stored in five-year categories.
- Within older age groups the model separates people less well (see AUC by group).
- No external validation set: the holdout is a random split of the same survey.
- Diabetes, sex and smoking codings follow the Kaggle documentation as understood by the
  author and should be re-checked against the data card.

Generated at: {m.get("trained_at")}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in (("model_a", card_a), ("model_b", card_b)):
        path = ARTIFACTS / name / "v1" / "metrics.json"
        if not path.exists():
            print(f"skip {name}: {path} not found (train it first)")
            continue
        metrics = json.loads(path.read_text())
        target = OUT / f"{name}.md"
        target.write_text(builder(metrics))
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
