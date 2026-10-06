"""
🔮 Predict — single customer churn prediction with a SHAP explanation.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from app_common import (
    categorize_risk,
    get_feature_options,
    load_all_models,
    load_raw_data,
    predict_single,
    render_customer_form,
)
from src import Config
from src.Explain import build_explainer, explain_customer

st.set_page_config(page_title="Predict — Telco Churn", page_icon="🔮", layout="wide")
st.title("🔮 Predict Churn for a Customer")
st.caption("Fill in a customer's details, then predict their churn risk and see why.")

try:
    models = load_all_models()
    df = load_raw_data()
except FileNotFoundError:
    st.error("⚠️ No trained model found. Run `python main.py` from the project root first.")
    st.stop()

options = get_feature_options(df)

# --- model + threshold controls ---
st.divider()
control_col1, control_col2 = st.columns([2, 1])
with control_col1:
    model_names = list(models.keys())
    default_index = model_names.index("XGBoost") if "XGBoost" in model_names else 0
    model_name = st.selectbox("Model to use", model_names, index=default_index)
with control_col2:
    threshold = st.slider(
        "Classification threshold", min_value=0.0, max_value=1.0,
        value=Config.DEFAULT_THRESHOLD, step=0.01,
        help=(
            "The model outputs a probability. This threshold decides where "
            "that probability becomes a Yes/No churn prediction — it doesn't "
            "change the probability itself, only how it's labeled."
        ),
    )

# --- input form ---
st.divider()
customer = render_customer_form(options)

# --- compact profile summary, not a dataframe dump ---
st.divider()
st.subheader("👤 Customer Profile")
p1, p2, p3 = st.columns(3)
p1.metric("Tenure", f"{customer['tenure']:.0f} months")
p1.metric("Contract", customer["Contract"])
p2.metric("Internet Service", customer["InternetService"])
p2.metric("Monthly Charges", f"${customer['MonthlyCharges']:.2f}")
p3.metric("Payment Method", customer["PaymentMethod"])
p3.metric("Paperless Billing", customer["PaperlessBilling"])

# --- predict ---
st.divider()
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Running model..."):
        model = models[model_name]
        proba = predict_single(model, customer)
        risk = categorize_risk(proba)
        predicted_label = "Churn" if proba >= threshold else "No Churn"

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        if risk == "High":
            st.error(f"🔴 **{risk} Churn Risk**")
        elif risk == "Medium":
            st.warning(f"🟠 **{risk} Churn Risk**")
        else:
            st.success(f"🟢 **{risk} Churn Risk**")
        st.caption(f"At threshold {threshold:.2f}, predicted class: **{predicted_label}**")
    with result_col2:
        st.metric("Churn Probability", f"{proba:.1%}")
        st.progress(proba)

    # --- SHAP explanation ---
    st.subheader("🔬 Why did the model make this prediction?")
    customer_df = pd.DataFrame([customer])
    shap_values, X_enc, feature_names = build_explainer(model, customer_df)
    fig, shap_proba, reconstructed = explain_customer(model, shap_values, customer_df, 0)
    st.pyplot(fig)

    with st.expander("🧑‍💻 Technical SHAP details"):
        st.write(f"Reconstructed probability from SHAP contributions: **{reconstructed:.3f}**")
        st.write(f"Model's own predict_proba: **{shap_proba:.3f}**")
        st.caption(
            "These should match almost exactly — SHAP values plus the "
            "baseline reconstruct the model's exact output. This describes "
            "how the model used each feature for *this* prediction; it is "
            "not proof of a causal relationship between any feature and churn."
        )

    # --- log to session history ---
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    st.session_state.prediction_history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Model": model_name,
        "Probability": f"{proba:.1%}",
        "Predicted": predicted_label,
        "Risk": risk,
        "Tenure": customer["tenure"],
        "Contract": customer["Contract"],
    })

# --- history table + download (persists across reruns via session_state) ---
if st.session_state.get("prediction_history"):
    st.divider()
    st.subheader("📋 Prediction History (this session)")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)
    st.download_button(
        "📥 Download history as CSV",
        data=history_df.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )