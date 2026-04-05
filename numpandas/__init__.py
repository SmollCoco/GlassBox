"""Minimal, production-focused pandas-like API using NumPy."""

from .core.dataframe import DataFrame
from .core.series import Series
from .core.index import Index
from .io.csv import read_csv
from .io.json import read_json
from .io.excel import read_excel
from .ml import (
    KNNClassifier,
    KNNRegressor,
    LinearRegressionGD,
    LogisticRegressionGD,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    train_test_split,
)

__all__ = [
    "DataFrame",
    "Series",
    "Index",
    "read_csv",
    "read_json",
    "read_excel",
    "LinearRegressionGD",
    "LogisticRegressionGD",
    "KNNClassifier",
    "KNNRegressor",
    "train_test_split",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]
