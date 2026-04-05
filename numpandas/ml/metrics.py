"""Evaluation metrics for simple ML workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import to_numpy_1d


def _binary_targets(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = to_numpy_1d(y_true, dtype=None, name="y_true")
    pred = to_numpy_1d(y_pred, dtype=None, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    labels = np.unique(np.concatenate([true, pred]))
    if labels.size > 2:
        raise ValueError("Binary classification metrics support only two classes.")
    return true, pred


def accuracy_score(y_true: Any, y_pred: Any) -> float:
    true = to_numpy_1d(y_true, dtype=None, name="y_true")
    pred = to_numpy_1d(y_pred, dtype=None, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(true == pred))


def precision_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float:
    true, pred = _binary_targets(y_true, y_pred)
    predicted_positive = pred == positive_label
    true_positive = np.sum((true == positive_label) & predicted_positive)
    false_positive = np.sum((true != positive_label) & predicted_positive)
    denom = true_positive + false_positive
    if denom == 0:
        return 0.0
    return float(true_positive / denom)


def recall_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float:
    true, pred = _binary_targets(y_true, y_pred)
    actual_positive = true == positive_label
    true_positive = np.sum((pred == positive_label) & actual_positive)
    false_negative = np.sum((pred != positive_label) & actual_positive)
    denom = true_positive + false_negative
    if denom == 0:
        return 0.0
    return float(true_positive / denom)


def f1_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float:
    precision = precision_score(y_true, y_pred, positive_label=positive_label)
    recall = recall_score(y_true, y_pred, positive_label=positive_label)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return float(2 * precision * recall / denom)


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    true = to_numpy_1d(y_true, name="y_true")
    pred = to_numpy_1d(y_pred, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean(np.abs(true - pred)))


def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    true = to_numpy_1d(y_true, name="y_true")
    pred = to_numpy_1d(y_pred, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return float(np.mean((true - pred) ** 2))


def r2_score(y_true: Any, y_pred: Any) -> float:
    true = to_numpy_1d(y_true, name="y_true")
    pred = to_numpy_1d(y_pred, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    total = np.sum((true - np.mean(true)) ** 2)
    if total == 0:
        return 0.0
    residual = np.sum((true - pred) ** 2)
    return float(1 - (residual / total))
