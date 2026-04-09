"""CSV I/O utilities."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np

from ..core.dataframe import DataFrame
from ..utils.dtypes import infer_dtype, parse_scalar


def read_csv(filepath: str, sep: str = ",", header: bool = True) -> DataFrame:
    """Read a CSV file into a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to CSV file.
    sep : str, default ','
        Column delimiter.
    header : bool, default True
        Whether the first row is a header.

    Returns
    -------
    DataFrame
        Parsed DataFrame.
    """
    with open(filepath, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter=sep)
        rows = list(reader)
    if not rows:
        return DataFrame({})
    if header:
        columns = rows[0]
        data_rows = rows[1:]
    else:
        columns = [str(idx) for idx in range(len(rows[0]))]
        data_rows = rows
    col_values = {name: [] for name in columns}
    for row in data_rows:
        if len(row) != len(columns):
            raise ValueError("Row length does not match header.")
        for name, value in zip(columns, row):
            col_values[name].append(parse_scalar(value))
    data = {}
    for name, values in col_values.items():
        dtype = infer_dtype(values)
        data[name] = np.array(values, dtype=dtype)
    return DataFrame(data, columns=columns)


def write_csv(frame: DataFrame, filepath: str, index: bool = False) -> None:
    """Write a DataFrame to CSV.

    Parameters
    ----------
    frame : DataFrame
        DataFrame to write.
    filepath : str
        Output path.
    index : bool, default False
        Whether to include the index.
    """
    with open(filepath, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = frame.columns
        if index:
            writer.writerow(["index"] + header)
        else:
            writer.writerow(header)
        for row_idx in range(len(frame.index)):
            row = [frame._data[col][row_idx] for col in frame.columns]
            if index:
                row = [frame.index.to_list()[row_idx]] + row
            writer.writerow(row)
