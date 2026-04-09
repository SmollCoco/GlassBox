"""Simple gradient-descent regression and classification models."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import accuracy_score, r2_score
from .utils import add_intercept, to_numpy_1d, to_numpy_2d, validate_same_length


class LinearRegressionGD:
    """Linear regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        tolerance: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.intercept_ = 0.0
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit(self, X: Any, y: Any) -> "LinearRegressionGD":
        X_array, y_array = validate_same_length(X, y)
        design = add_intercept(X_array) if self.fit_intercept else X_array
        weights = np.zeros(design.shape[1], dtype=float)
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            predictions = design @ weights
            errors = predictions - y_array
            loss = float(np.mean(errors ** 2))
            self.loss_history_.append(loss)
            gradient = (design.T @ errors) / design.shape[0]
            new_weights = weights - self.learning_rate * gradient
            if np.linalg.norm(new_weights - weights) <= self.tolerance:
                weights = new_weights
                break
            weights = new_weights

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict.")
        X_array = to_numpy_2d(X, name="X")
        return X_array @ self.coef_ + self.intercept_

    def score(self, X: Any, y: Any) -> float:
        return r2_score(y, self.predict(X))


class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        tolerance: float = 1e-8,
        threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.tolerance = tolerance
        self.threshold = threshold
        self.intercept_ = 0.0
        self.coef_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: Any, y: Any) -> "LogisticRegressionGD":
        X_array, y_array = validate_same_length(X, y)
        unique_labels = np.unique(y_array)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("LogisticRegressionGD supports binary targets encoded as 0 and 1.")

        design = add_intercept(X_array) if self.fit_intercept else X_array
        weights = np.zeros(design.shape[1], dtype=float)
        self.loss_history_ = []

        for _ in range(self.n_iterations):
            logits = design @ weights
            probabilities = self._sigmoid(logits)
            clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
            loss = float(-np.mean(y_array * np.log(clipped) + (1 - y_array) * np.log(1 - clipped)))
            self.loss_history_.append(loss)
            gradient = (design.T @ (probabilities - y_array)) / design.shape[0]
            new_weights = weights - self.learning_rate * gradient
            if np.linalg.norm(new_weights - weights) <= self.tolerance:
                weights = new_weights
                break
            weights = new_weights

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.copy()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Model must be fit before calling predict_proba.")
        X_array = to_numpy_2d(X, name="X")
        positive = self._sigmoid(X_array @ self.coef_ + self.intercept_)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def score(self, X: Any, y: Any) -> float:
        return accuracy_score(y, self.predict(X))
