# Model card: Model B (Lifestyle check)

## Intended use
Educational demonstration of an explainable screening-style estimate from answers a
person can give without any tests. Not a medical device. Not for medical decisions.

## Data
CDC BRFSS 2015 survey, cleaned version distributed on Kaggle. 229781 rows after
removing duplicates. Target: respondent *reported ever having* coronary heart disease or a
heart attack (10.3% of rows). Survey sampling weights
are not included, so rates describe this file, not the US population.

## Features (8)
HighBP, HighChol, BMI, Smoker, Diabetes, PhysActivity, Sex, Age

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
- 5-fold cross-validation AUC: 0.796 (std 0.003)
- Stratified 20% holdout (calibrated): AUC 0.796, average precision
  0.289, Brier 0.082

By group (holdout):

| Group | People | Reported disease | AUC | Avg. precision | Brier |
| --- | --- | --- | --- | --- | --- |
| sex=0 | 25718 | 2039 | 0.785 | 0.22 | 0.0669 |
| sex=1 | 20239 | 2704 | 0.79 | 0.33 | 0.1013 |
| age 18-39 | 6930 | 68 | 0.725 | 0.048 | 0.0096 |
| age 40-59 | 16465 | 942 | 0.759 | 0.165 | 0.0508 |
| age 60-74 | 16206 | 2311 | 0.724 | 0.288 | 0.1126 |
| age 75+ | 6356 | 1422 | 0.67 | 0.351 | 0.1639 |

Result groups (cut-offs [0.05, 0.15]), observed on the holdout:

| Result group | People | Reported disease | Share of people |
| --- | --- | --- | --- |
| lower | 19653 | 0.019 | 0.428 |
| moderate | 14967 | 0.095 | 0.326 |
| higher | 11337 | 0.26 | 0.247 |

## Known limitations
- Self-reported, cross-sectional data: the model describes who *reports* heart disease,
  it does not predict future disease.
- US adults in 2015; not validated elsewhere. Age is stored in five-year categories.
- Within older age groups the model separates people less well (see AUC by group).
- No external validation set: the holdout is a random split of the same survey.
- Diabetes, sex and smoking codings follow the Kaggle documentation as understood by the
  author and should be re-checked against the data card.

Generated at: 2026-10-22T14:19:23+00:00
