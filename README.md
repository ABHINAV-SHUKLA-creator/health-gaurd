# HealthGuard - explainable heart-disease risk screening

A full-stack machine-learning project: two models trained from scratch, an API that
explains every prediction, and a website to use them.

> **Educational project. Not a medical device and not medical advice.**
> Nothing here is validated for real medical decisions.

## What it does

| | Clinical check (Model A) | Lifestyle check (Model B) |
|---|---|---|
| Data | UCI Heart Disease, 4 hospitals, 920 patients | CDC BRFSS 2015 survey (Kaggle version), ~230k respondents |
| Inputs | age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, exercise angina | age, sex, height/weight (BMI), smoking, physical activity, high BP, high cholesterol, diabetes |
| Model | calibrated logistic regression + SHAP | calibrated logistic regression + SHAP |
| Output | risk group + relative score + per-answer explanation | risk group + share of similar respondents reporting heart disease + explanation |
| Honest evaluation | leave-one-hospital-out testing | stratified holdout, subgroup (sex/age) results |

The point of the project is **interpretability and honest evaluation**, not a headline accuracy number.

## Key findings (see `docs/model_cards/` for full numbers)

- **Model A** reaches pooled cross-validation AUC of about 0.87, but when a whole hospital is held out,
  AUC ranges from about 0.75 to 0.88. Predicted scores are off from actual disease rates by up to
  30 points at some hospitals, so the site shows a *relative* score and a risk group, not a "% chance".
- **Model B** reaches AUC of about 0.80 and is already well calibrated. It is weaker among the oldest
  age group. Risk groups separate people clearly (observed rates rise steeply across groups).
- A simple logistic regression matched or beat random forest and XGBoost on both datasets.
- Features were removed on purpose and the reasons are documented: specialist-test columns
  (not enterable on a form), outcome-like columns (stroke, general health), income/insurance,
  and heavy-alcohol (a misleading protective sign).

## Architecture

```
Browser (frontend/: HTML + CSS + JS)
        |  JSON over HTTP
FastAPI (backend/app)  --  /predict/clinical, /predict/lifestyle, /models/*/info
        |
artifacts/model_a/v1, artifacts/model_b/v1   (model.joblib, explainer.joblib, metrics.json, plots)
        ^
ml/scripts/*.py  (data cleaning, experiments, training)
```

One FastAPI process serves both the API and the website.

## Run it locally (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
run_site.bat
```

Open http://127.0.0.1:8000 for the site and http://127.0.0.1:8000/docs for the interactive API docs.
The trained models are included in `artifacts/`, so no retraining is needed to run the site.

## Reproduce the models

```bat
python data\download.py                 :: UCI files
:: BRFSS: download the CSV from Kaggle (see data/README.md) into data\raw\brfss\
python ml\scripts\prepare_data.py
python ml\scripts\baseline_a.py         :: model comparison for Model A
python ml\scripts\ablation_a.py         :: feature-set ablation for Model A
python ml\scripts\train_model_a.py
python ml\scripts\baseline_b.py
python ml\scripts\prune_b.py
python ml\scripts\train_model_b.py
python ml\scripts\make_model_cards.py   :: regenerates docs/model_cards from the saved metrics
```

Seeds are fixed, so reruns give the same numbers.

## Tests and linting

```bat
flake8 ml data backend
python -m pytest ml\tests
cd backend && python -m pytest tests
```

GitHub Actions runs the same checks on every push (`.github/workflows/ci.yml`).

## Deploy

A `Dockerfile` is included (it serves the API and website on `$PORT`):

```bash
docker build -t healthguard .
docker run -p 8000:8000 healthguard
```

Any host that runs Docker containers from a GitHub repo (for example Render) can deploy it directly.
The trained models in `artifacts/` must be committed to the repository.

## Project layout

```
data/        download + cleaning inputs (raw and processed data are git-ignored)
ml/          experiments, training, evaluation, model-card generator
artifacts/   trained models, explainers, metrics and plots
backend/     FastAPI app and tests
frontend/    the website (no build step)
docs/        generated model cards
```

## Data sources and licences

- UCI Heart Disease (Cleveland, Hungarian, Switzerland, VA Long Beach), CC BY 4.0.
- CDC BRFSS 2015 as distributed on Kaggle ("Heart Disease Health Indicators", by Alex Teboul).
  Check the licence on the Kaggle data card before redistributing.

## Limitations

Small and old clinical data; self-reported US survey data; no testing outside these datasets;
scores are relative (Model A) or describe self-reported history, not future risk (Model B).
See the model cards for details.
