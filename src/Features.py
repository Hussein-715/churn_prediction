"""
Preprocessing: turns raw customer data into the numeric matrix a model needs.

Mirrors notebook step 10. Numeric columns pass through unchanged (trees need
no scaling); text columns are one-hot encoded.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build (but do not fit) a ColumnTransformer sized to X's columns."""
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )
