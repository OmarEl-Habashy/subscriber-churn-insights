# Subscriber Churn Insights

Churn risk scoring and retention insights for a subscription streaming service —
event-log ingestion, behavioral feature engineering, model comparison, SHAP
explainability, and a live scoring API + dashboard.

**Status:** API is live with the trained model wired in. Dashboard deployed;
KPI/risk-table wiring to real data in progress.

- **Live API:** https://subscriber-churn-api.onrender.com (`/docs` for interactive spec)
- **Live dashboard:** https://subscriber-churn-dashboard.onrender.com

> Free-tier hosting — services sleep after ~15 min idle and take 30–60s to wake
> on the first request.

## Problem

A streaming platform loses subscribers to churn. Flagging at-risk users before
they cancel lets a retention team intervene with targeted offers or
re-engagement pushes before it's too late. This project scores churn risk per
user, explains *why* each user is at risk, and serves those scores through a
live API and dashboard.

## Results

Model selection was done via 5-fold stratified cross-validation, comparing
Logistic Regression, Linear SVM, and XGBoost on identical folds (not a single
train/test split — with 225 users, a single split is too small to trust; see
`src/train.py` for the full comparison).

| Model | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| Logistic Regression | 0.75 ± 0.10 | 0.92 ± 0.05 | 0.85 ± 0.05 |
| Linear SVM | 0.76 ± 0.04 | 0.91 ± 0.04 | 0.85 ± 0.02 |
| **XGBoost (selected)** | **0.78 ± 0.15** | **0.91 ± 0.06** | **0.86 ± 0.07** |

XGBoost was selected on mean PR-AUC (the most reliable metric here given
class imbalance — ~23% churn rate). Note it also has the highest variance
across folds; Linear SVM is a close, more stable runner-up.

## Architecture

```
Raw event logs → clean → feature engineering → 5-fold CV model comparison
(LR / SVM / XGBoost) → SHAP explainability → FastAPI inference service → Streamlit dashboard
```

## Stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit. Deployed on Render.

## Repo layout

```
api/          FastAPI service (POST /predict, GET /health)
dashboard/    Streamlit dashboard
src/          shared feature engineering, training, and explainability code
artifacts/    trained model + scaler + feature schema
notebooks/    EDA and experimentation
```

## Running locally

```bash
pip install -r requirements.txt

# 1. build the feature table (requires the Sparkify mini dataset in data/)
python -m src.features

# 2. train + evaluate (5-fold CV, saves artifacts/)
python -m src.train data/sparkify_features.csv

# 3. API
uvicorn api.main:app --reload --port 8000

# 4. Dashboard (separate terminal)
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```

