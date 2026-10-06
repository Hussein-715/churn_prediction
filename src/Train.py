"""
Splitting the data and tuning each model.

Each `tune_*` function searches over a full Pipeline (preprocessor +
classifier) at once, using scikit-learn's ``step__param`` naming
(e.g. ``classifier__max_depth``). Tuning the pipeline directly — rather
than tuning the bare classifier on pre-encoded data and wrapping it in a
Pipeline afterward — means `search.best_estimator_` comes out already a
complete, fitted Pipeline, ready to evaluate and save with no extra
"does this match the manual version" check needed.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import Config
from .Features import build_preprocessor


def split_data(df: pd.DataFrame, target_col: str = Config.TARGET_COLUMN):
    """Stratified 80/20 train/test split, matching notebook step 9."""
    X = df.drop(columns=target_col)
    y = df[target_col]
    return train_test_split(
        X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE
    )


def make_cv() -> StratifiedKFold:
    """The single CV splitter shared by every model, so all three are
    tuned and compared on identical folds."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)


def tune_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> GridSearchCV:
    """Grid search over a Decision Tree pipeline. Mirrors notebook step 17."""
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", DecisionTreeClassifier(random_state=Config.RANDOM_STATE)),
    ])
    param_grid = {
        "classifier__max_depth": [3, 4, 5, 6, 8, 10, None],
        "classifier__min_samples_leaf": [1, 5, 10, 20, 50],
        "classifier__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> RandomizedSearchCV:
    """Randomized search over a Random Forest pipeline. Mirrors notebook step 19."""
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=1)),
    ])
    param_dist = {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [None, 10, 15, 20],
        "classifier__min_samples_leaf": [1, 5, 10, 20],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": [None, "balanced"],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=20, scoring="f1", cv=cv,
        random_state=Config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold) -> RandomizedSearchCV:
    """Randomized search over an XGBoost pipeline. Mirrors notebook step 21."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=Config.RANDOM_STATE,
            n_jobs=1,
        )),
    ])
    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 4, 5, 6, 8],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__scale_pos_weight": [1, scale_pos_weight],
    }
    search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=20, scoring="f1", cv=cv,
        random_state=Config.RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search


