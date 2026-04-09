"""Gaussian Naive Bayes for classification."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import to_numpy_2d, to_numpy_1d


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.

    Assumes each feature follows a Gaussian distribution within each class.
    Uses log-probabilities to avoid numerical underflow.

    Parameters
    ----------
    var_smoothing : float
        Added to all variances for numerical stability (like epsilon).
        Default mirrors sklearn's 1e-9.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative.")
        self.var_smoothing = var_smoothing

        self.classes_: np.ndarray | None = None
        self._log_priors: np.ndarray | None = None  # shape: (n_classes,)
        self._means: np.ndarray | None = None  # shape: (n_classes, n_features)
        self._vars: np.ndarray | None = None  # shape: (n_classes, n_features)
        self.n_features_: int | None = None

    # ── public API ──────────────────────────────

    def fit(self, X: Any, y: Any) -> "GaussianNaiveBayes":
        X_arr = to_numpy_2d(X, name="X")
        y_arr = to_numpy_1d(y, dtype=None, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")

        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X_arr.shape
        self.n_features_ = n_features
        n_classes = len(self.classes_)

        self._means = np.zeros((n_classes, n_features))
        self._vars = np.zeros((n_classes, n_features))
        self._log_priors = np.zeros(n_classes)

        for i, c in enumerate(self.classes_):
            X_c = X_arr[y_arr == c]
            self._means[i] = X_c.mean(axis=0)
            # Keep variances strictly positive to avoid log/divide by zero.
            self._vars[i] = (
                np.maximum(X_c.var(axis=0), np.finfo(float).eps) + self.var_smoothing
            )
            self._log_priors[i] = np.log(len(X_c) / n_samples)

        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = to_numpy_2d(X, name="X")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
            )
        log_posteriors = self._compute_log_posteriors(X_arr)  # (n_samples, n_classes)
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities via softmax on log-posteriors.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        self._check_fitted()
        X_arr = to_numpy_2d(X, name="X")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
            )
        log_posteriors = self._compute_log_posteriors(X_arr)
        # numerically stable softmax per row
        log_posteriors -= log_posteriors.max(axis=1, keepdims=True)
        exp_vals = np.exp(log_posteriors)
        return exp_vals / exp_vals.sum(axis=1, keepdims=True)

    # ── internals ───────────────────────────────

    def _compute_log_posteriors(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(y | X) ∝ log P(y) + Σ log P(xᵢ | y) for each class.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            # Gaussian log-likelihood: -0.5 * [log(2πσ²) + (x-μ)²/σ²]
            log_likelihood = -0.5 * (
                np.log(2 * np.pi * self._vars[i])  # (n_features,)
                + (X - self._means[i]) ** 2 / self._vars[i]  # (n_samples, n_features)
            )
            log_posteriors[:, i] = self._log_priors[i] + log_likelihood.sum(axis=1)

        return log_posteriors

    def _check_fitted(self) -> None:
        if (
            self.classes_ is None
            or self._log_priors is None
            or self._means is None
            or self._vars is None
            or self.n_features_ is None
        ):
            raise RuntimeError("Call fit() before predict().")
