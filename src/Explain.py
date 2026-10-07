"""
SHAP explanations for a fitted pipeline.

Mirrors notebook steps 26-30. Works with any Pipeline that has a
"preprocessor" step and a tree-based "classifier" step (Decision Tree,
Random Forest and XGBoost are all supported by shap.TreeExplainer).
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.pipeline import Pipeline


def build_explainer(pipeline: Pipeline, X: pd.DataFrame):
    """Build a TreeExplainer for the classifier inside `pipeline`.

    Returns (shap_values, X_encoded, feature_names). SHAP explains the
    classifier directly, so X is passed through the pipeline's own
    preprocessor first, exactly like notebook step 26.
    """
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_enc = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_enc)

    # scikit-learn classifiers (Decision Tree, Random Forest) expose a
    # 2-column predict_proba, and TreeExplainer mirrors that with one set
    # of SHAP values PER CLASS: shape (n_samples, n_features, n_classes).
    # XGBoost's binary:logistic objective has a single internal output,
    # so its SHAP values come back as (n_samples, n_features) with no
    # class dimension at all. Keep only the churn (class 1) slice when a
    # class dimension exists, so every caller downstream — the waterfall
    # plot, the summary plot, the reconstruction check — can treat all
    # three model types identically and never needs to know this quirk
    # exists.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values.feature_names = list(feature_names)

    return shap_values, X_enc, feature_names


def plot_global_importance(shap_values, X_enc, feature_names):
    """Bar chart (mean |SHAP value|) and beeswarm (direction). Matches
    notebook step 27. Returns (bar_fig, beeswarm_fig) instead of calling
    plt.show() — see explain_customer's note below for why."""
    shap.summary_plot(shap_values, X_enc, feature_names=feature_names, plot_type="bar", show=False)
    bar_fig = plt.gcf()
    bar_fig.axes[0].set_title("Global feature importance — mean |SHAP value|")
    plt.tight_layout()

    shap.summary_plot(shap_values, X_enc, feature_names=feature_names, show=False)
    beeswarm_fig = plt.gcf()
    plt.tight_layout()

    return bar_fig, beeswarm_fig


def most_at_risk_customer(pipeline: Pipeline, X: pd.DataFrame) -> int:
    """Position (not the pandas index label) of the customer with the
    highest predicted churn probability."""
    proba = pipeline.predict_proba(X)[:, 1]
    return int(np.argmax(proba))


def explain_customer(pipeline: Pipeline, shap_values, X: pd.DataFrame, index: int):
    """Waterfall plot for one customer, plus the numeric check that
    base_value + sum(contributions), through a sigmoid, reproduces the
    pipeline's own predicted probability. Matches notebook steps 28-30.
    `index` is a position into X (as from most_at_risk_customer), not a
    pandas index label.

    Returns (fig, proba, reconstructed_proba) instead of printing and
    calling plt.show() — a console can do `print(...)` / `plt.show(fig)`
    with these, and the Streamlit app can do `st.metric(...)` /
    `st.pyplot(fig)` with the exact same values.
    """
    proba = float(pipeline.predict_proba(X)[:, 1][index])

    shap.plots.waterfall(shap_values[index], show=False)
    fig = plt.gcf()
    plt.tight_layout()

    raw_score = shap_values[index].base_values + shap_values[index].values.sum()
    reconstructed = float(expit(raw_score))

    return fig, proba, reconstructed
