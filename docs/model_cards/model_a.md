# Model card: Model A (Clinical check)

## Intended use
Educational demonstration of explainable risk estimation from basic checkup values.
Not a medical device. Not for diagnosis, treatment or screening decisions.

## Data
UCI Heart Disease, four sites (Cleveland, Hungarian, Switzerland, VA Long Beach),
920 patients (CC BY 4.0). Target: any angiographic disease (original
label greater than 0). Zero values for cholesterol and blood pressure were treated as
missing and filled with the median.

## Features (7)
age, sex, cp, trestbps, chol, fbs, exang

## Model
Standardised logistic regression (C = 0.1), sigmoid-calibrated, with exact SHAP
(linear explainer) for per-person explanations. Chosen over random forest and XGBoost
because it matched them on held-out hospitals and is simpler to explain.

## Performance
- Pooled 5-fold cross-validation AUC: 0.866
  (std 0.016)
- Pooled held-out-hospital AUC: 0.84; Brier 0.162
  (always predicting the base rate would score about 0.247)

Leave-one-hospital-out (train on three hospitals, test on the fourth):

| Held-out hospital | Patients | Actual disease rate | Mean predicted score | AUC | Brier |
| --- | --- | --- | --- | --- | --- |
| cleveland | 303 | 0.459 | 0.576 | 0.847 | 0.178 |
| hungarian | 294 | 0.361 | 0.472 | 0.882 | 0.145 |
| switzerland | 123 | 0.935 | 0.652 | 0.783 | 0.159 |
| va | 200 | 0.745 | 0.707 | 0.752 | 0.164 |

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

Generated at: 2026-10-21T07:06:25+00:00
