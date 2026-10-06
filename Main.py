"""
End-to-end pipeline: load -> clean -> split -> tune all three models ->
compare on the test set -> save the best one, with metadata.

This is the .py equivalent of the whole exploration notebook, runnable
in one command instead of cell by cell.

Run from the project root:
    python main.py
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import sklearn

from src import Config
from src.Data import load_and_clean
from src.Evaluate import compare_models, select_best_model
from src.Train import make_cv, split_data, tune_decision_tree, tune_random_forest, tune_xgboost


def _json_safe(value):
    """Convert numpy scalar types to plain Python types so json.dump
    doesn't choke on them (the same issue you hit saving the notebook's
    metadata in step 52 — best_params_ can contain numpy types)."""
    return value.item() if hasattr(value, "item") else value


def main() -> None:
    print("Loading and cleaning data...")
    df = load_and_clean()
    X_train, X_test, y_train, y_test = split_data(df)
    cv = make_cv()

    print("\nTuning Decision Tree...")
    tree_search = tune_decision_tree(X_train, y_train, cv)
    print(f"  best CV F1: {tree_search.best_score_:.3f}")

    print("\nTuning Random Forest...")
    forest_search = tune_random_forest(X_train, y_train, cv)
    print(f"  best CV F1: {forest_search.best_score_:.3f}")

    print("\nTuning XGBoost...")
    xgb_search = tune_xgboost(X_train, y_train, cv)
    print(f"  best CV F1: {xgb_search.best_score_:.3f}")

    searches = {
        "Decision Tree": tree_search,
        "Random Forest": forest_search,
        "XGBoost": xgb_search,
    }
    models = {name: search.best_estimator_ for name, search in searches.items()}

    print("\nComparing on the held-out test set...")
    comparison = compare_models(models, X_test, y_test)
    print(comparison)

    best_name, best_model = select_best_model(models, comparison)
    print(f"\nBest model: {best_name}")

    Config.MODELS_DIR.mkdir(exist_ok=True)

    # Save every model individually, so the app can load any of them, not
    # just whichever one happened to win.
    for name, model in models.items():
        joblib.dump(model, Config.MODELS_DIR / Config.MODEL_FILENAMES[name])

    # Plus a convenience copy of the overall best, under a stable filename,
    # for anything that just wants "the recommended model" without caring
    # which one that is.
    joblib.dump(best_model, Config.MODEL_PATH)

    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": best_name,
        "best_params": {k: _json_safe(v) for k, v in searches[best_name].best_params_.items()},
        "test_set_metrics": comparison.loc[best_name].to_dict(),
        "model_comparison": comparison.to_dict(orient="index"),
        "random_state": Config.RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    with open(Config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=_json_safe)

    print(f"\nSaved model to    {Config.MODEL_PATH}")
    print(f"Saved metadata to {Config.METADATA_PATH}")


if __name__ == "__main__":
    main()