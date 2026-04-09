"""Decision Tree for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import to_numpy_2d, to_numpy_1d


# ──────────────────────────────────────────────
# Internal node structure
# ──────────────────────────────────────────────


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(
        self,
        *,
        feature: int | None = None,
        threshold: float | None = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        value: Any = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set only on leaf nodes

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ──────────────────────────────────────────────
# Decision Tree
# ──────────────────────────────────────────────


class DecisionTree:
    """Decision Tree supporting both classification and regression.

    Parameters
    ----------
    task : {"classification", "regression"}
    max_depth : int | None
        Maximum tree depth. None = grow until pure/min_samples.
    min_samples_split : int
        Minimum samples required to attempt a split.
    """

    def __init__(
        self,
        task: str = "classification",
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be a positive integer or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: _Node | None = None
        self.n_features_: int | None = None

    # ── public API ──────────────────────────────

    def fit(self, X: Any, y: Any) -> "DecisionTree":
        X_arr = to_numpy_2d(X, name="X")
        if self.task == "classification":
            y_arr = to_numpy_1d(y, dtype=None, name="y")
        else:
            y_arr = to_numpy_1d(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")
        self.n_features_ = X_arr.shape[1]
        self._root = self._build(X_arr, y_arr, depth=0)
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = to_numpy_2d(X, name="X")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
            )
        return np.array([self._traverse(x, self._root) for x in X_arr])

    # ── split criteria ───────────────────────────

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        """Gini impurity for a label array."""
        n = len(y)
        if n == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return float(1.0 - np.sum(probs**2))

    @staticmethod
    def _mse(y: np.ndarray) -> float:
        """Variance (MSE from mean) for a value array."""
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self.task == "classification" else self._mse(y)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None, float]:
        """Return (best_feature, best_threshold, best_gain)."""
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = -np.inf
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]
                n_l, n_r = len(y_left), len(y_right)

                # weighted impurity reduction
                gain = parent_impurity - (
                    n_l / n_samples * self._impurity(y_left)
                    + n_r / n_samples * self._impurity(y_right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)

        return best_feature, best_threshold, best_gain

    # ── tree construction ────────────────────────

    def _leaf_value(self, y: np.ndarray) -> Any:
        if self.task == "classification":
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]
        return float(np.mean(y))

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)

        # stop conditions → leaf
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            return _Node(value=self._leaf_value(y))

        feature, threshold, gain = self._best_split(X, y)

        if feature is None or gain <= 0:
            return _Node(value=self._leaf_value(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[right_mask], y[right_mask], depth + 1),
        )

    # ── prediction traversal ─────────────────────

    def _traverse(self, x: np.ndarray, node: _Node) -> Any:
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    # ── helpers ──────────────────────────────────

    def _check_fitted(self) -> None:
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")
