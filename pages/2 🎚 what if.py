"""
🎚️ What-If — see how changing tenure or contract type alone shifts a
customer's predicted churn risk, holding everything else at a typical
("baseline") value.
"""

import pandas as pd
import streamlit as st

from app_common import (
    categorize_risk,
    get_baseline_customer,
    get_feature_options,
    load_all_models,
    load_raw_data,
    predict_single,
)

st.set_page_config(page_title="What-If — Telco Churn", page_icon="🎚️", layout="wide")
st.title("🎚️ Model What-If Analysis")
st.caption(
    "Starting from a typical customer, see how changing tenure or "
    "contract type alone shifts the model's predicted churn risk."
)

try:
    models = load_all_models()
    df = load_raw_data()
except FileNotFoundError:
    st.error("⚠️ No trained model found. Run `python main.py` from the project root first.")
    st.stop()

options = get_feature_options(df)
baseline = get_baseline_customer(df)

model_names = list(models.keys())
default_index = model_names.index("XGBoost") if "XGBoost" in model_names else 0
model_name = st.selectbox("Model to use", model_names, index=default_index)
model = models[model_name]

baseline_proba = predict_single(model, baseline)

st.divider()
st.caption(
    "Every field below except Tenure and Contract is held at the "
    "dataset's typical value (median for numbers, most common category "
    "for text) — shown here so it's clear what's being varied and what isn't."
)
with st.expander("📋 Baseline customer (everything held constant)"):
    rows = [(k, f"{v:.1f}" if isinstance(v, float) else v) for k, v in baseline.items()]
    st.table(pd.DataFrame(rows, columns=["Feature", "Baseline value"]))

st.divider()
st.subheader("Adjust the two levers")
lever_col1, lever_col2 = st.columns(2)
with lever_col1:
    tenure_val = st.slider(
        "Tenure (months)",
        min_value=int(options["tenure"]["min"]), max_value=int(options["tenure"]["max"]),
        value=int(baseline["tenure"]),
    )
with lever_col2:
    contract_options = options["Contract"]["options"]
    default_contract_index = (
        contract_options.index(baseline["Contract"]) if baseline["Contract"] in contract_options else 0
    )
    contract_val = st.selectbox("Contract", contract_options, index=default_contract_index)

whatif = dict(baseline)
whatif["tenure"] = float(tenure_val)
whatif["Contract"] = contract_val

whatif_proba = predict_single(model, whatif)
delta_pp = (whatif_proba - baseline_proba) * 100

st.divider()
st.subheader("🧪 Result")
res1, res2, res3 = st.columns(3)
res1.metric("Baseline probability", f"{baseline_proba:.1%}")
res2.metric("New probability", f"{whatif_proba:.1%}", delta=f"{delta_pp:+.1f} pp", delta_color="inverse")
res3.metric("Risk category", categorize_risk(whatif_proba))

st.info(
    "This simulation shows how the model's *prediction* changes when "
    "these inputs are modified — it does not establish that changing a "
    "real customer's tenure or contract would causally produce this "
    "change in their actual likelihood of churning.",
    icon="🧪",
)

# --- sensitivity chart: probability vs tenure, one line per contract type ---
st.divider()
st.subheader("📈 Sensitivity: churn probability vs tenure, by contract")
tenure_range = list(range(int(options["tenure"]["min"]), int(options["tenure"]["max"]) + 1))
chart_rows = []
for contract_type in contract_options:
    scenario = dict(baseline)
    scenario["Contract"] = contract_type
    for t in tenure_range:
        scenario["tenure"] = float(t)
        chart_rows.append({
            "Tenure": t, "Contract": contract_type,
            "Churn probability": predict_single(model, scenario),
        })

chart_df = pd.DataFrame(chart_rows).pivot(index="Tenure", columns="Contract", values="Churn probability")
st.line_chart(chart_df)
st.caption(
    "Each line holds every feature except tenure and contract type at "
    "the baseline value above. This is not a forecast for any specific "
    "customer — it's a view of how the model's output shifts across one "
    "feature at a time."
)