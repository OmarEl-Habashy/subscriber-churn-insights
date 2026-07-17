"""
Subscriber Churn Insights — inference API.

STATUS: skeleton. /health is real. /predict currently returns a rule-based
placeholder so the service is genuinely live end-to-end while the trained
model is still being built. Swap the placeholder logic in `predict_stub`
for `model.predict_proba(...)` once artifacts/model.json exists — see
the TODO below.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Subscriber Churn Insights API",
    version="0.1.0",
    description="Churn risk scoring for subscriber behavioral data.",
)


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


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_tier: str
    top_factors: list[dict]
    model_status: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "subscriber-churn-insights-api"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: UserFeatures):
    # TODO: replace with real model inference once artifacts/model.json is trained:
    #   prob = model.predict_proba(scaler.transform([feature_vector]))[0][1]
    #   shap_values = explainer(feature_vector)
    prob = predict_stub(features)
    tier = "high" if prob >= 0.6 else "medium" if prob >= 0.3 else "low"

    return PredictionResponse(
        churn_probability=round(prob, 3),
        risk_tier=tier,
        top_factors=[
            {"feature": "num_thumb_down", "impact": "placeholder", "direction": "n/a"},
        ],
        model_status="placeholder — trained model not yet deployed",
    )


def predict_stub(f: UserFeatures) -> float:
    """Simple heuristic standing in for the trained model. Not a real score."""
    risk = 0.3
    if f.num_thumb_down > f.num_thumb_up:
        risk += 0.2
    if f.listen_time < 5000:
        risk += 0.15
    if f.tot_friends == 0:
        risk += 0.1
    return min(risk, 0.95)
