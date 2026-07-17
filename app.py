"""
Subscriber Churn Insights — retention dashboard.

STATUS: skeleton. Calls the live FastAPI service (placeholder /predict for
now). Replace the sample_users list with real scored users once batch
scoring / the trained model is wired up.
"""

import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Retention Console", layout="wide")

st.title("Retention Console")
st.caption("Subscriber churn risk monitoring — skeleton build")

# --- API status check ---
try:
    resp = requests.get(f"{API_URL}/health", timeout=5)
    api_status = "online" if resp.status_code == 200 else "🔴 unreachable"
except requests.exceptions.RequestException:
    api_status = "unreachable"

st.sidebar.markdown(f"**API status:** {api_status}")
st.sidebar.markdown(f"`{API_URL}`")

st.warning(
    "This is a skeleton deployment. Model, KPIs, and the risk table below are "
    "placeholders until the trained model is wired in.",
    icon="",
)

# --- placeholder KPI row ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active users scored", "—")
col2.metric("Predicted churn rate", "—")
col3.metric("High-risk users", "—")
col4.metric("Model AUC-ROC", "—")

st.subheader("Try the API")
st.write("Send a sample user's features to the live `/predict` endpoint:")

sample_payload = {
    "gender": 1,
    "tot_songs": 342,
    "listen_time": 88210.5,
    "num_thumb_up": 12,
    "num_thumb_down": 3,
    "add_to_playlist": 5,
    "tot_friends": 4,
    "lt": 1209600000,
    "avg_played_songs": 28.4,
    "tot_artist_played": 190,
}

if st.button("Score sample user"):
    try:
        r = requests.post(f"{API_URL}/predict", json=sample_payload, timeout=10)
        st.json(r.json())
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
