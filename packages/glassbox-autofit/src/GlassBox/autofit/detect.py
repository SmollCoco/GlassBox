from __future__ import annotations

import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame


def detect_task(df: DataFrame, target_col: str) -> str:
    """Detect whether the target indicates classification or regression."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the DataFrame.")

    target = df[target_col].to_numpy()
    dtype = df.dtypes[target_col]

    if np.issubdtype(dtype, np.bool_):
        return "classification"

    if np.issubdtype(dtype, np.integer):
        unique_count = int(np.unique(target).shape[0])
        if unique_count <= 10:
            return "classification"
        return "regression"

    if np.issubdtype(dtype, np.floating):
        return "regression"

    if dtype == np.dtype("O"):
        return "classification"

    raise ValueError(
        f"Unsupported target dtype '{dtype}' for automatic task detection."
    )
