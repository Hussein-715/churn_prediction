"""
Shared loading and helper functions for the Streamlit app.

Every page imports from here instead of repeating model/data loading, so
there's one cached copy of each model and the dataset for the whole app
session — not one reload per page, and not one reload per widget click.
"""

import json
from typing import Dict, Tuple

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src import Config
from src.Data import load_and_clean
from src.Train import split_data


@st.cache_resource
def load_model(name: str) -> Pipeline:
    """Load one fitted pipeline by name (a key of config.MODEL_FILENAMES).

    Cached with `st.cache_resource`, not `st.cache_data`: a fitted Pipeline
    is a complex object (not plain data), so we want Streamlit to keep the
    exact same instance in memory across reruns rather than trying to hash
    or copy it.
    """
    path = Config.MODELS_DIR / Config.MODEL_FILENAMES[name]
    return joblib.load(path)


@st.cache_resource
def load_all_models() -> Dict[str, Pipeline]:
    """Load all three fitted pipelines, keyed by model name."""
    return {name: load_model(name) for name in Config.MODEL_FILENAMES}


@st.cache_data
def load_metadata() -> dict:
    """Load the metadata JSON saved by main.py: best hyperparameters, test
    metrics, and the full model comparison table. Cached with
    `st.cache_data`, since a plain dict is safe for Streamlit to hash."""
    with open(Config.METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load and clean the full dataset. Used to build input-widget options
    from real values, and to recreate the test split below."""
    return load_and_clean()


@st.cache_data
def get_test_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Recreate the exact train/test split used during training (same
    `random_state`), so the app's live metrics match the notebook's
    exactly, with no retraining needed."""
    df = load_raw_data()
    return split_data(df)


def get_feature_options(df: pd.DataFrame) -> dict:
    """Build the choices for every input widget directly from the real
    dataset, so the form can never offer a category or range the model
    has never seen.

    Returns {column: {"type": "numeric", "min", "max", "median"}} for
    numeric columns, or {"type": "categorical", "options": [...]} for
    text columns.
    """
    options = {}
    feature_cols = df.drop(columns=Config.TARGET_COLUMN).columns
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            options[col] = {
                "type": "numeric",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }
        else:
            options[col] = {
                "type": "categorical",
                "options": sorted(df[col].dropna().unique().tolist()),
            }
    return options


def categorize_risk(probability: float) -> str:
    """Map a churn probability to a risk label, using the app-defined
    boundaries in config.RISK_THRESHOLDS.

    Independent of the classification threshold (which decides the
    Yes/No label) — this is a separate, finer-grained display category,
    and always uses the raw probability regardless of where the
    threshold slider is set.
    """
    if probability < Config.RISK_THRESHOLDS["low"]:
        return "Low"
    elif probability < Config.RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def predict_single(pipeline: Pipeline, customer: dict) -> float:
    """Predict the churn probability for one customer, given as a dict of
    {column_name: value}. Builds the one-row DataFrame the pipeline
    expects internally."""
    customer_df = pd.DataFrame([customer])
    return float(pipeline.predict_proba(customer_df)[:, 1][0])


# --- Input form building blocks, shared by the Predict and What-If pages ---

# Groups the 19 feature columns into the same 3 sections used throughout
# the project's EDA, so a 19-field form reads as 3 short ones.
FEATURE_GROUPS = {
    "👤 Customer Information": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"],
    "📡 Services": [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ],
    "💳 Account & Billing": ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
}

# Human-readable widget labels for the raw column names.
FEATURE_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior Citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure (months)",
    "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method",
    "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
}


def render_feature_input(col: str, meta: dict, key_suffix: str = ""):
    """Render the right widget for one feature and return its current
    value, already in the type the model expects.

    This is where input validation actually lives: a numeric widget's
    min/max come straight from the real training data, so a value the
    model has never seen is impossible to enter — not caught afterward.

    `SeniorCitizen` is a special case: stored as 0/1 in the data, but
    clearer to a person as a Yes/No choice.
    """
    label = FEATURE_LABELS.get(col, col)
    key = f"{col}_{key_suffix}" if key_suffix else col

    if col == "SeniorCitizen":
        choice = st.selectbox(label, ["No", "Yes"], key=key)
        return 1 if choice == "Yes" else 0

    if meta["type"] == "numeric":
        step = 1.0 if col == "tenure" else 0.5
        help_text = (
            "Typically close to tenure × Monthly Charges for this customer."
            if col == "TotalCharges" else None
        )
        return st.number_input(
            label, min_value=meta["min"], max_value=meta["max"],
            value=meta["median"], step=step, help=help_text, key=key,
        )

    return st.selectbox(label, meta["options"], key=key)


def get_baseline_customer(df: pd.DataFrame) -> dict:
    """Build a 'typical' customer from the dataset: median for numeric
    columns, mode (most common value) for text columns. Used as the fixed
    baseline in the What-If page — only the levers a user explicitly
    adjusts differ from this baseline, so the effect being shown is
    attributable to that one change."""
    customer = {}
    feature_cols = df.drop(columns=Config.TARGET_COLUMN).columns
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            customer[col] = float(df[col].median())
        else:
            customer[col] = df[col].mode().iloc[0]
    return customer


def render_customer_form(options: dict, key_suffix: str = "") -> dict:
    """Render the full 3-section input form and return the collected
    customer as a dict, ready for predict_single / build_explainer.

    `key_suffix` keeps widget keys unique when the same form is rendered
    more than once on one page (e.g. "original" vs "modified" in the
    What-If page) — without it, Streamlit would treat two forms with
    identical labels as the same widget.
    """
    customer = {}
    for section_title, cols in FEATURE_GROUPS.items():
        st.subheader(section_title)
        widget_cols = st.columns(3)
        for i, col in enumerate(cols):
            with widget_cols[i % 3]:
                customer[col] = render_feature_input(col, options[col], key_suffix=key_suffix)
    return customer