"""
Evaluating and comparing the tuned pipelines on the held-out test set.

Mirrors notebook steps 18, 20 and 22-23. Every function here takes an
already-fitted pipeline (e.g. ``search.best_estimator_`` from train.py) and
the untouched test set.
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, name: str = "Model") -> dict:
    """Score one fitted pipeline on the test set. Prints the full
    classification report and returns the summary metrics as a dict."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    return {
        "Precision (Churn)": precision_score(y_test, y_pred),
        "Recall (Churn)": recall_score(y_test, y_pred),
        "F1 (Churn)": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC": average_precision_score(y_test, y_proba),
    }


def compare_models(models: Dict[str, BaseEstimator], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate every model and return one comparison table, matching
    notebook step 23."""
    rows = {name: evaluate_model(model, X_test, y_test, name=name) for name, model in models.items()}
    return pd.DataFrame(rows).T.round(3)


def select_best_model(
    models: Dict[str, BaseEstimator], comparison: pd.DataFrame, metric: str = "F1 (Churn)"
) -> Tuple[str, BaseEstimator]:
    """Pick the model with the highest score on `metric`. F1 on the
    Churned class is the default, since it's the metric the project
    settled on over plain accuracy — pass a different `metric` to
    optimize for something else (e.g. "Recall (Churn)")."""
    best_name = comparison[metric].idxmax()
    return best_name, models[best_name]


def plot_confusion_matrix(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series, title: str = "Confusion matrix"):
    """Returns the Figure instead of calling plt.show(), so a notebook can
    call plt.show(fig) and the Streamlit app can call st.pyplot(fig) —
    same function, two different places to render it."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(cmap="Blues")
    disp.ax_.set_title(title)
    return disp.figure_


def plot_roc_pr_curves(models: Dict[str, BaseEstimator], X_test: pd.DataFrame, y_test: pd.Series):
    """Overlay ROC and Precision-Recall curves for every model, matching
    the comparison plot from notebook step 23. Returns the Figure rather
    than calling plt.show() — see plot_confusion_matrix's note above."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=name)
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name=name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_title("ROC curves — all models")
    axes[1].set_title("Precision-Recall curves — all models")
    plt.tight_layout()
    return fig