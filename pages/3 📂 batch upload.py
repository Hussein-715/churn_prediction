"""
📂 Batch Upload — score a CSV of many customers at once, ranked by
predicted churn risk.
"""

import pandas as pd
import streamlit as st

from app_common import categorize_risk, get_feature_options, load_all_models, load_raw_data
from src.Data import prepare_for_inference

st.set_page_config(page_title="Batch Upload — Telco Churn", page_icon="📂", layout="wide")
st.title("📂 Batch Upload")
st.caption("Upload a CSV of customers and score all of them at once, ranked by churn risk.")

try:
    models = load_all_models()
    training_df = load_raw_data()
except FileNotFoundError:
    st.error("⚠️ No trained model found. Run `python main.py` from the project root first.")
    st.stop()

expected_cols = set(get_feature_options(training_df).keys())

model_names = list(models.keys())
default_index = model_names.index("XGBoost") if "XGBoost" in model_names else 0
model_name = st.selectbox("Model to use", model_names, index=default_index)
model = models[model_name]

st.divider()

with st.expander("📄 Need a template?"):
    template_df = training_df.drop(columns="Churn").head(3)
    st.dataframe(template_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download a 3-row template CSV",
        data=template_df.to_csv(index=False),
        file_name="customer_template.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader(
    "Upload a CSV of customers",
    type="csv",
    help="Same columns as the training data. customerID is optional (used only to label results); Churn, if present, is ignored.",
)

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

# --- read, with a friendly error instead of a raw traceback on bad input ---
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"⚠️ Couldn't read that file as a CSV. Details: {e}")
    st.stop()

if raw_df.empty:
    st.error("⚠️ That CSV has no rows.")
    st.stop()

# Keep an identifier for the leaderboard before prepare_for_inference drops
# it — the model doesn't need it, but the results are only actionable if
# each row can still be traced back to a customer.
if "customerID" in raw_df.columns:
    customer_ids = raw_df["customerID"].astype(str)
else:
    customer_ids = pd.Series([f"Row {i}" for i in range(len(raw_df))], name="customerID")

try:
    prepared_df = prepare_for_inference(raw_df)
except Exception as e:
    st.error(f"⚠️ Something went wrong while preparing this data. Details: {e}")
    st.stop()

missing_cols = expected_cols - set(prepared_df.columns)
if missing_cols:
    st.error("⚠️ This CSV is missing columns the model needs: " + ", ".join(sorted(missing_cols)))
    st.stop()

try:
    with st.spinner(f"Scoring {len(prepared_df):,} customers..."):
        probabilities = model.predict_proba(prepared_df)[:, 1]
except Exception as e:
    st.error(f"⚠️ Something went wrong while scoring these customers. Details: {e}")
    st.stop()

risk_labels = [categorize_risk(p) for p in probabilities]

results_df = pd.DataFrame({
    "customerID": customer_ids.values,
    "Churn Probability": probabilities,
    "Risk": risk_labels,
    "Tenure": prepared_df["tenure"].values,
    "Contract": prepared_df["Contract"].values,
    "Monthly Charges": prepared_df["MonthlyCharges"].values,
}).sort_values("Churn Probability", ascending=False).reset_index(drop=True)

# --- summary ---
st.divider()
st.subheader("📊 Summary")
risk_counts = results_df["Risk"].value_counts()
sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
sum_col1.metric("Customers scored", f"{len(results_df):,}")
sum_col2.metric("🔴 High risk", f"{risk_counts.get('High', 0):,}")
sum_col3.metric("🟠 Medium risk", f"{risk_counts.get('Medium', 0):,}")
sum_col4.metric("🟢 Low risk", f"{risk_counts.get('Low', 0):,}")

# --- leaderboard ---
st.divider()
st.subheader("🏆 Risk Leaderboard")
display_df = results_df.copy()
display_df["Churn Probability"] = display_df["Churn Probability"].map(lambda p: f"{p:.1%}")
display_df["Monthly Charges"] = display_df["Monthly Charges"].map(lambda c: f"${c:.2f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    "📥 Download full results as CSV",
    data=results_df.to_csv(index=False),
    file_name="churn_risk_leaderboard.csv",
    mime="text/csv",
)