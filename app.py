"""
Home page. Streamlit auto-builds the sidebar navigation from the files in
pages/ — this script is just what's shown when the app first loads.

Run with:
    streamlit run app.py
"""

import streamlit as st

from app_common import load_metadata, load_raw_data

st.set_page_config(page_title="Telco Customer Churn", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction")
st.markdown(
    "Predict which customers are likely to churn, understand *why* the model "
    "thinks so, and compare the three models this project trained."
)

# Friendly error instead of a raw traceback if main.py hasn't been run yet
try:
    metadata = load_metadata()
    df = load_raw_data()
except FileNotFoundError:
    st.error(
        "⚠️ No trained model found. Run `python main.py` from the project "
        "root first — it trains all three models and saves them, which "
        "this app needs before it can predict anything."
    )
    st.stop()

st.divider()

st.success(f"🏆 **Active model:** {metadata['model_type']}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total customers", f"{len(df):,}")
with col2:
    churn_rate = df["Churn"].mean()
    st.metric("Churn rate", f"{churn_rate:.1%}")
with col3:
    f1 = metadata["test_set_metrics"]["F1 (Churn)"]
    st.metric("F1 (Churn)", f"{f1:.3f}")
with col4:
    roc_auc = metadata["test_set_metrics"]["ROC-AUC"]
    st.metric("ROC-AUC", f"{roc_auc:.3f}")

st.caption(
    f"Metrics are from the held-out test set, saved when the model was "
    f"last trained ({metadata.get('saved_at_utc', 'unknown date')[:10]})."
)

st.divider()

st.subheader("What's in this app")
st.markdown(
    "- **🔮 Predict** — enter a customer's details and get a churn "
    "probability, risk category, and a SHAP explanation of why\n"
    "- **🎚️ What-If** — see how changing one feature shifts a customer's "
    "predicted risk\n"
    "- **📂 Batch Upload** — score a whole CSV of customers at once, "
    "ranked by risk\n"
    "- **📊 Compare Models** — the real test-set comparison across all "
    "three trained models"
)

st.info(
    "**How to read this app:** churn probabilities are statistical "
    "estimates from patterns in historical data, not certainties. SHAP "
    "explanations describe what the model used to make its prediction — "
    "they are not proof of what *causes* a customer to churn. What-if "
    "results show how the model's output changes, not what would "
    "actually happen if that customer's plan changed.",
    icon="ℹ️",
)

st.caption("Use the sidebar to navigate between pages.")