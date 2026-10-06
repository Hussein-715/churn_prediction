"""
Central configuration for the churn prediction project.

Every other module imports its constants from here instead of hardcoding
paths or magic numbers — change a value once, it takes effect everywhere.
"""

from pathlib import Path

# Reproducibility: passed to every random process (train/test split,
# model training, cross-validation folds) so results are repeatable.
RANDOM_STATE = 42

# Paths — resolved relative to this file, so the project runs correctly
# no matter which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"       # convenience copy of the overall best model
METADATA_PATH = MODELS_DIR / "churn_pipeline_metadata.json"

# Every tuned model is saved individually (not just the winner), so the
# app can let a user switch between them or compare them side by side.
MODEL_FILENAMES = {
    "Decision Tree": "decision_tree_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
}

# Train/test split
TEST_SIZE = 0.2

# Target column
TARGET_COLUMN = "Churn"

# --- App-level settings (not used in training — these only affect how the
# Streamlit app turns a probability into a decision) ---

# The probability above which a prediction counts as "churn". 0.5 is the
# conventional default; the app lets the user adjust this, since the
# right threshold depends on the cost of a false negative vs a false
# positive, not on the model itself.
DEFAULT_THRESHOLD = 0.5

# Risk category boundaries. These are application-defined judgment calls
# for display purposes only, not a property of the model or the dataset.
RISK_THRESHOLDS = {
    "low": 0.30,     # probability < 0.30  -> Low risk
    "medium": 0.60,  # 0.30 <= probability < 0.60 -> Medium risk
                      # probability >= 0.60 -> High risk
}