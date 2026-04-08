"""IO helpers for numpandas."""

from .csv import read_csv
from .json import read_json
from .excel import read_excel

__all__ = ["read_csv", "read_json", "read_excel"]
