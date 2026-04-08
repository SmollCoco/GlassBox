"""Helpers for the lightweight ML module."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_2d(data: Any, *, dtype: Any = float, name: str = "X") -> np.ndarray:
    """Convert supported inputs to a 2D NumPy array."""
    if hasattr(data, "to_numpy"):
        array = data.to_numpy()
    else:
        array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D structure.")
    try:
        array = array.astype(dtype, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")
    return array


def to_numpy_1d(data: Any, *, dtype: Any | None = float, name: str = "y") -> np.ndarray:
    """Convert supported inputs to a 1D NumPy array."""
    if hasattr(data, "to_numpy"):
        array = data.to_numpy()
    else:
        array = np.asarray(data)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        array = array.reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if dtype is not None:
        try:
            array = array.astype(dtype, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} has an invalid dtype.") from exc
    if np.issubdtype(array.dtype, np.number) and np.isnan(array).any():
        raise ValueError(f"{name} must not contain NaN values.")
    return array


def validate_same_length(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate feature and target lengths."""
    X_array = to_numpy_2d(X, name="X")
    y_array = to_numpy_1d(y, name="y")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    return X_array, y_array


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Append a bias column to the feature matrix."""
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([ones, X])


def train_test_split(
    X: Any,
    y: Any,
    test_size: float = 0.2,
    random_state: int | None = None,
    shuffle: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and test subsets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    X_array, y_array = validate_same_length(X, y)
    n_samples = X_array.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    test_count = max(1, int(round(n_samples * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    if train_indices.size == 0:
        raise ValueError("test_size leaves no training samples.")
    return (
        X_array[train_indices],
        X_array[test_indices],
        y_array[train_indices],
        y_array[test_indices],
    )
