"""JSON I/O utilities."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..core.dataframe import DataFrame
from ..utils.dtypes import infer_dtype, is_nan_value, parse_scalar


def _to_json_scalar(value: Any) -> Any:
    """Convert NumPy scalars to JSON-serializable Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_json(filepath: str) -> DataFrame:
    """Read a JSON file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to JSON file.

    Returns
    -------
    DataFrame
        Parsed DataFrame.
    """
    with open(filepath, "r") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if not payload:
            return DataFrame({})
        columns = list(payload[0].keys())
        data = {name: [] for name in columns}
        for row in payload:
            for name in columns:
                data[name].append(parse_scalar(row.get(name)))
    elif isinstance(payload, dict):
        columns = list(payload.keys())
        data = {name: [parse_scalar(item) for item in payload[name]] for name in columns}
    else:
        raise ValueError("Unsupported JSON structure.")
    typed = {}
    for name, values in data.items():
        dtype = infer_dtype(values)
        typed[name] = np.array(values, dtype=dtype)
    return DataFrame(typed, columns=columns)


def write_json(frame: DataFrame, filepath: str) -> None:
    """Write a DataFrame to JSON file.

    Parameters
    ----------
    frame : DataFrame
        DataFrame to write.
    filepath : str
        Output path.
    """
    records = []
    for row_idx in range(len(frame.index)):
        row = {}
        for col in frame.columns:
            value = frame._data[col][row_idx]
            if is_nan_value(value):
                row[col] = None
            else:
                row[col] = _to_json_scalar(value)
        records.append(row)
    with open(filepath, "w") as handle:
        json.dump(records, handle)
