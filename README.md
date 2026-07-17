# Subscriber Churn Insights

Churn risk scoring and retention insights for a subscription streaming service —
event-log ingestion, behavioral feature engineering, model comparison, SHAP
explainability, and a live scoring API + dashboard.



## Problem

A streaming platform loses subscribers to churn. Flagging at-risk users before
they cancel lets a retention team intervene with targeted offers or
re-engagement pushes before it's too late. This project scores churn risk per
user, explains *why* each user is at risk, and serves those scores through a
live API and dashboard.

## Architecture

```
Raw event logs → clean → feature engineering → train (LR baseline + XGBoost)
→ SHAP explainability → FastAPI inference service → Streamlit dashboard
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

# API
uvicorn api.main:app --reload --port 8000

# Dashboard (separate terminal)
API_URL=http://localhost:8000 streamlit run dashboard/app.py
```
