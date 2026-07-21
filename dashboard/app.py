import json
import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLE_USERS_PATH = os.path.join(os.path.dirname(__file__), "sample_users.json")

st.set_page_config(page_title="Retention Console", layout="wide")

st.title("Retention Console")
st.caption("Subscriber churn risk monitoring — live model")

try:
    resp = requests.get(f"{API_URL}/health", timeout=10)
    health = resp.json() if resp.status_code == 200 else {}
    api_online = resp.status_code == 200
    model_loaded = health.get("model_loaded", False)
    model_name = health.get("model_name", "—")
except requests.exceptions.RequestException:
    api_online = False
    model_loaded = False
    model_name = "—"

status_label = "🟢 online" if api_online else "🔴 unreachable"
st.sidebar.markdown(f"**API status:** {status_label}")
st.sidebar.markdown(f"**Model:** `{model_name}`" if model_loaded else "**Model:** not loaded")
st.sidebar.markdown(f"`{API_URL}`")

if not api_online:
    st.error(
        "Can't reach the scoring API right now. Free-tier services sleep after "
        "idle time — try refreshing in ~30-60s, or check the API's /health endpoint directly.",
        icon="⚠️",
    )
    st.stop()

if not model_loaded:
    st.warning(
        "API is online but the trained model artifacts aren't loaded — "
        "check that artifacts/ was committed and the API redeployed.",
        icon="🚧",
    )
    st.stop()


@st.cache_data(ttl=300, show_spinner="Scoring sample users against the live model...")
def score_sample_users(api_url: str):
    with open(SAMPLE_USERS_PATH) as f:
        users = json.load(f)

    scored = []
    for user in users:
        user_id = user.pop("user_id")
        try:
            r = requests.post(f"{api_url}/predict", json=user, timeout=15)
            r.raise_for_status()
            result = r.json()
            scored.append({
                "user_id": user_id,
                "churn_probability": result["churn_probability"],
                "risk_tier": result["risk_tier"],
                "top_factors": result["top_factors"],
                "tenure_days": round(user["lt"] / (1000 * 60 * 60 * 24), 1),
            })
        except requests.exceptions.RequestException:
            continue
    return scored


scored_users = score_sample_users(API_URL)

if not scored_users:
    st.error("No users could be scored — check the API logs.")
    st.stop()

df_scored = pd.DataFrame(scored_users).sort_values("churn_probability", ascending=False)

churn_rate = (df_scored["risk_tier"] != "low").mean()
high_risk_count = (df_scored["risk_tier"] == "high").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Users scored (sample)", len(df_scored))
col2.metric("At-risk rate", f"{churn_rate:.0%}")
col3.metric("High-risk users", int(high_risk_count))
col4.metric("Model", model_name.upper())

st.caption(
    "KPIs computed from a sample of 20 real historical users scored live against "
    "the deployed model — not the full dataset, and not synthetic data."
)

# --- risk table ---
st.subheader("At-risk users")

for _, row in df_scored.iterrows():
    tier_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[row["risk_tier"]]
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 3])
        with c1:
            st.markdown(f"**user_{row['user_id']}**")
            st.caption(f"tenure {row['tenure_days']:.0f}d")
        with c2:
            st.markdown(f"{tier_color} **{row['risk_tier'].upper()}**")
            st.caption(f"p = {row['churn_probability']:.2f}")
        with c3:
            chips = []
            for factor in row["top_factors"]:
                arrow = "↑" if factor["direction"] == "increases risk" else "↓"
                chips.append(f"`{arrow} {factor['feature']}`")
            st.markdown(" ".join(chips))

st.divider()
st.caption(
    "Try scoring a custom user directly against the API: "
    f"[{API_URL}/docs]({API_URL}/docs)"
)