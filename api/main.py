import json
import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Subscriber Churn Insights API",
    version="0.1.1",
    description="Churn risk scoring for subscriber behavioral data.",
)

try:
    model = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    with open("artifacts/feature_columns.json") as f:
        FEATURE_COLUMNS = json.load(f)
    with open("artifacts/model_info.json") as f:
        MODEL_INFO = json.load(f)
    NEEDS_SCALING = MODEL_INFO.get("needs_scaling", False)

    #use SHAP explainer
    explainer = shap.TreeExplainer(model) if not NEEDS_SCALING else shap.Explainer(model, scaler.transform)
    ARTIFACTS_LOADED = True

except FileNotFoundError : #not crash 
    model = scaler = FEATURE_COLUMNS = MODEL_INFO = explainer = None
    NEEDS_SCALING = False
    ARTIFACTS_LOADED = False



class UserFeatures(BaseModel):
    gender: int = Field(..., description="1 = F, 0 = M")
    tot_songs: int
    listen_time: float
    num_thumb_up: int
    num_thumb_down: int
    add_to_playlist: int
    tot_friends: int
    lt: float = Field(..., description="lifetime / tenure in ms")
    avg_played_songs: float
    tot_artist_played: int
    days_since_last_session: float
    downgrade_flag: int = Field(..., description="1 if user ever downgraded, else 0")
    negative_ratio: float = Field(..., description="thumbs_down / (thumbs_up + thumbs_down + 1)")
    sessions_per_week: float
    ads_per_session: float


class Factor(BaseModel):
    feature: str
    impact: float
    direction: str

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str
    top_factors: list[Factor]
    model_status: str


@app.get("/health")
def health():
    return {
        "status" : "ok",
        "service" : "subscriber-churn-insights-api",
        "model_loaded" : ARTIFACTS_LOADED,
        "model_name" : MODEL_INFO.get("model_name") if MODEL_INFO else None,

    }


@app.post("/predict", response_model = PredictionResponse)
def predict(features: UserFeatures):
    if not ARTIFACTS_LOADED:
        raise HTTPException(
            status_code=503,
            detail = "Model Artifacts not found. redeploy!"
        )
    row= [getattr(features, col) for col in FEATURE_COLUMNS]
    X = np.array([row], dtype=float)
    X_input = scaler.transform(X) if NEEDS_SCALING else X

    prob = float(model.predict_proba(X_input)[0][1])
    tier = "high" if prob >= 0.6 else "medium" if prob >= 0.3 else "low"

    shap_values = explainer(X_input)
    values = shap_values.values[0]
    factor_pairs = sorted(
        zip(FEATURE_COLUMNS, values), key=lambda p: abs(p[1]), reverse=True
    )[:3]
    top_factors = [
        Factor(
            feature=name,
            impact=round(float(val), 4),
            direction="increases risk" if val > 0 else "decreases risk",
        )
        for name, val in factor_pairs
    ]
 
    return PredictionResponse(
        churn_probability=round(prob, 3),
        risk_tier=tier,
        top_factors=top_factors,
        model_status=f"live — {MODEL_INFO.get('model_name', 'unknown')} (5-fold CV validated)",
    )