"""
📊 Compare Models — live comparison of all three tuned models on the
held-out test set, plus a closer look at whichever one you select.
"""

import streamlit as st

from app_common import get_test_split, load_all_models, load_metadata
from src.Evaluate import compare_models, plot_confusion_matrix, plot_roc_pr_curves, select_best_model

st.set_page_config(page_title="Compare Models — Telco Churn", page_icon="📊", layout="wide")
st.title("📊 Compare Models")
st.caption(
    "Live metrics on the same held-out test set used throughout this "
    "project — recomputed from the saved models, not stored numbers."
)

try:
    models = load_all_models()
    load_metadata()  # just confirms training actually completed
except FileNotFoundError:
    st.error("⚠️ No trained model found. Run `python main.py` from the project root first.")
    st.stop()

X_train, X_test, y_train, y_test = get_test_split()

with st.spinner("Evaluating all three models on the test set..."):
    comparison = compare_models(models, X_test, y_test)

best_name, best_model = select_best_model(models, comparison)

st.divider()
st.success(f"🏆 **Overall best model (by F1 on the Churned class):** {best_name}")

st.subheader("Comparison table")
display_comparison = comparison.copy()
for col in display_comparison.columns:
    display_comparison[col] = display_comparison[col].map(lambda v: f"{v:.3f}")
st.dataframe(display_comparison, use_container_width=True)
st.caption(
    "F1, ROC-AUC and PR-AUC are prioritized over plain accuracy here — "
    "with about 73% of customers not churning, a model that always "
    "predicts 'stayed' would already look misleadingly accurate."
)

# --- inspect one model ---
st.divider()
st.subheader("Inspect one model")

KEY_HYPERPARAMS = {
    "Decision Tree": ["max_depth", "min_samples_leaf", "class_weight"],
    "Random Forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight"],
    "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight"],
}

model_names = list(models.keys())
selected_name = st.selectbox("Model to inspect", model_names, index=model_names.index(best_name))
selected_model = models[selected_name]

insp_col1, insp_col2 = st.columns(2)
with insp_col1:
    st.markdown("**Confusion matrix**")
    cm_fig = plot_confusion_matrix(selected_model, X_test, y_test, title=selected_name)
    st.pyplot(cm_fig)

with insp_col2:
    st.markdown("**Model information**")
    classifier = selected_model.named_steps["classifier"]
    preprocessor = selected_model.named_steps["preprocessor"]
    params = classifier.get_params()

    st.metric("Training samples", f"{len(X_train):,}")
    st.metric("Features (after encoding)", f"{len(preprocessor.get_feature_names_out())}")

    for p in KEY_HYPERPARAMS.get(selected_name, []):
        if p in params:
            st.write(f"**{p}:** {params[p]}")

    with st.expander("🧑‍💻 View all technical details"):
        # Hyperparameters can include numpy scalar types (e.g. scale_pos_weight,
        # computed via pandas) — convert to plain Python types first, the same
        # fix applied in main.py's metadata export, for the same reason.
        safe_params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        st.json(safe_params)

st.caption(
    "These are the model's actual saved hyperparameters, read directly "
    "from the fitted pipeline — not retyped by hand, so they can't drift "
    "out of sync with what's actually running."
)

# --- ROC / PR overlay across all three ---
st.divider()
st.subheader("ROC and Precision-Recall curves — all models")
curves_fig = plot_roc_pr_curves(models, X_test, y_test)
st.pyplot(curves_fig)
st.caption(
    "ROC-AUC measures how well a model ranks churners above stayers "
    "across every possible threshold. PR-AUC focuses specifically on the "
    "churned (minority) class, making it the more informative of the two "
    "given the class imbalance in this data."
)