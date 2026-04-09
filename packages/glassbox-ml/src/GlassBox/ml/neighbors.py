"""K-nearest neighbors models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import accuracy_score, r2_score
from .utils import to_numpy_1d, to_numpy_2d, validate_same_length


class _BaseKNN:
    def __init__(self, n_neighbors: int = 5, distance_metric: str = "euclidean"):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        if distance_metric not in {"euclidean", "manhattan"}:
            raise ValueError("distance_metric must be 'euclidean' or 'manhattan'.")
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: Any, y: Any):
        X_array, y_array = validate_same_length(X, y)
        if self.n_neighbors > X_array.shape[0]:
            raise ValueError("n_neighbors cannot exceed the number of training samples.")
        self._X_train = X_array
        self._y_train = y_array
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        if self._X_train is None:
            raise ValueError("Model must be fit before prediction.")
        diffs = X[:, np.newaxis, :] - self._X_train[np.newaxis, :, :]
        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum(diffs ** 2, axis=2))
        return np.sum(np.abs(diffs), axis=2)

    def _neighbor_targets(self, X: Any) -> np.ndarray:
        if self._X_train is None or self._y_train is None:
            raise ValueError("Model must be fit before prediction.")
        X_array = to_numpy_2d(X, name="X")
        distances = self._compute_distances(X_array)
        neighbor_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        return self._y_train[neighbor_indices]


class KNNRegressor(_BaseKNN):
    """Simple KNN regressor."""

    def predict(self, X: Any) -> np.ndarray:
        neighbors = self._neighbor_targets(X).astype(float)
        return np.mean(neighbors, axis=1)

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class KNNClassifier(_BaseKNN):
    """Simple KNN classifier."""

    def predict(self, X: Any) -> np.ndarray:
        neighbors = self._neighbor_targets(X)
        predictions = []
        for row in neighbors:
            labels, counts = np.unique(row, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))
