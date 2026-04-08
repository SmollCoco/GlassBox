"""Minimal, production-focused pandas-like API using NumPy."""

from .core.dataframe import DataFrame
from .core.series import Series
from .core.index import Index
from .io.csv import read_csv
from .io.json import read_json
from .io.excel import read_excel


__all__ = [
    "DataFrame",
    "Series",
    "Index",
    "read_csv",
    "read_json",
    "read_excel",
]
