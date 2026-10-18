# Data sources

## Model A - UCI Heart Disease (CC BY 4.0)
Hospitals: Cleveland, Hungarian, Switzerland, VA Long Beach.
https://archive.ics.uci.edu/dataset/45/heart-disease
Target: num is 0-4; we use target = (num > 0).
Cleaning: '?' -> missing; chol == 0 and trestbps == 0 -> missing
(placeholder zeros, verify in EDA); `site` column kept for
leave-one-hospital-out testing.

## Model B - BRFSS heart disease health indicators (Kaggle)
https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
Download the CSV manually and put it in data/raw/brfss/
The target records existing heart disease/attack, not future risk.
Check the dataset license on its Kaggle page before publishing.

Do not mix the Kaggle "Heart Failure Prediction" file with the raw UCI
files: it is built from the same patients and would leak across splits.