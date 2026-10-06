"""
Tests for src/data.py's cleaning logic.

Run from the project root with: pytest
"""

import pandas as pd
import pytest

from src.Data import clean_data


def _make_raw_df(**overrides) -> pd.DataFrame:
    """A minimal 1-row DataFrame with the real dataset's columns, so each
    test only has to specify the columns it cares about."""
    base = {
        "customerID": "0000-TEST",
        "tenure": 5,
        "TotalCharges": "100.50",
        "Churn": "No",
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_total_charges_is_converted_to_numeric():
    df = clean_data(_make_raw_df(TotalCharges="100.50"))
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].iloc[0] == 100.50


def test_blank_total_charges_filled_with_zero_when_tenure_is_zero():
    df = clean_data(_make_raw_df(tenure=0, TotalCharges=" "))
    assert df["TotalCharges"].iloc[0] == 0.0


def test_blank_total_charges_with_nonzero_tenure_raises():
    """This is the assumption verified in the notebook: a blank
    TotalCharges should only ever occur for a brand-new customer
    (tenure == 0). If that stops being true, cleaning should fail loudly
    rather than silently filling in a wrong number."""
    with pytest.raises(AssertionError):
        clean_data(_make_raw_df(tenure=12, TotalCharges=" "))


def test_customer_id_is_dropped():
    df = clean_data(_make_raw_df())
    assert "customerID" not in df.columns


def test_churn_is_mapped_to_binary():
    df_yes = clean_data(_make_raw_df(Churn="Yes"))
    df_no = clean_data(_make_raw_df(Churn="No"))
    assert df_yes["Churn"].iloc[0] == 1
    assert df_no["Churn"].iloc[0] == 0


def test_clean_data_does_not_mutate_input():
    raw = _make_raw_df()
    raw_copy = raw.copy()
    clean_data(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)