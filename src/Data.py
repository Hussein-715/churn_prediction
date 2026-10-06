"""
Loading and cleaning the raw Telco Customer Churn dataset.

These functions are the notebook's cleaning steps (sections 4-5), moved
here so training, testing and the future app all call the exact same
logic instead of three copies of it.
"""

from pathlib import Path

import pandas as pd

from . import Config


def load_raw_data(path: Path = Config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV. Raises a clear error if the file is missing,
    instead of letting pandas fail with a less obvious traceback."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from Kaggle "
            "(blastchar/telco-customer-churn) and place it there."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions verified in the notebook.

    Returns a new DataFrame; never modifies the input in place, so callers
    can safely keep a reference to the original raw data if they need it.
    """
    df = df.copy()

    # TotalCharges: stored as text because a few rows are blank. Verified
    # in the notebook that every blank belongs to a brand-new customer
    # (tenure == 0), so their true total is 0, not "unknown".
    total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = total_numeric.isna()
    assert (df.loc[blank_mask, "tenure"] == 0).all(), (
        "Found a blank TotalCharges for a customer with tenure > 0 — "
        "the fill-with-0 assumption no longer holds for this data, "
        "investigate before proceeding."
    )
    df["TotalCharges"] = total_numeric.fillna(0.0)

    # customerID: unique per row, nothing for a model to learn from it.
    df = df.drop(columns="customerID")

    # Churn: No -> 0, Yes -> 1, so metrics have a clear positive class.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def load_and_clean(path: Path = Config.DATA_PATH) -> pd.DataFrame:
    """Convenience wrapper: load then clean in one call."""
    return clean_data(load_raw_data(path))


def prepare_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer rows for *prediction*, not training.

    Deliberately lighter than `clean_data`: inference-time customers are
    unlabeled by definition, so there is no `Churn` column to map, and we
    don't assert the tenure==0 assumption, since we can't afford to crash
    a batch prediction over one unexpected row the way we can during
    training-data validation. `TotalCharges` and `customerID` are handled
    the same way either way.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    if "customerID" in df.columns:
        df = df.drop(columns="customerID")

    return df