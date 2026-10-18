# HealthGuard

Explainable heart-disease risk estimation with two models:

- Model A (Clinical Check): UCI Heart Disease, 4 hospitals
- Model B (Lifestyle Check): CDC BRFSS survey indicators

Educational project. Not a medical device. Not medical advice.

Setup (Windows):
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements-ml.txt
    python data\download.py
    python ml\scripts\prepare_data.py

Data sources and cleaning steps: see data/README.md