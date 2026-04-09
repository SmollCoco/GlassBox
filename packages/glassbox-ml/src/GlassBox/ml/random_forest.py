"""Random Forest for classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import to_numpy_2d, to_numpy_1d
from .decision_tree import DecisionTree


class RandomForest:
    """Random Forest ensemble using bagging + feature subspace sampling.

    Parameters
    ----------
    task : {"classification", "regression"}
    n_estimators : int
        Number of trees to build.
    max_depth : int | None
        Passed to each DecisionTree.
    min_samples_split : int
        Passed to each DecisionTree.
    max_features : int | float | {"sqrt", "log2"} | None
        Features to consider at each split.
        - int   → exact count
        - float → fraction of total features
        - "sqrt" (default) → floor(√n_features)
        - "log2" → floor(log2(n_features))
        - None  → all features (no subsampling)
    random_state : int | None
    """

    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        max_features: int | float | str | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be a positive integer or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self._trees: list[tuple[DecisionTree, np.ndarray]] = (
            []
        )  # (tree, feature_indices)
        self.n_features_: int | None = None

    # ── public API ──────────────────────────────

    def fit(self, X: Any, y: Any) -> "RandomForest":
        X_arr = to_numpy_2d(X, name="X")
        if self.task == "classification":
            y_arr = to_numpy_1d(y, dtype=None, name="y")
        else:
            y_arr = to_numpy_1d(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample.")

        n_samples, n_features = X_arr.shape
        self.n_features_ = n_features
        k = self._resolve_max_features(n_features)

        rng = np.random.default_rng(self.random_state)
        self._trees = []

        for _ in range(self.n_estimators):
            # bagging: sample rows with replacement
            row_idx = rng.integers(0, n_samples, size=n_samples)
            # feature subspace: sample k features without replacement
            col_idx = rng.choice(n_features, size=k, replace=False)
            col_idx = np.sort(col_idx)

            X_boot = X_arr[np.ix_(row_idx, col_idx)]
            y_boot = y_arr[row_idx]

            tree = DecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_boot, y_boot)
            self._trees.append((tree, col_idx))

        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_fitted()
        X_arr = to_numpy_2d(X, name="X")
        if X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X_arr.shape[1]}."
            )

        # collect each tree's predictions on its feature subset
        all_preds = np.stack(
            [tree.predict(X_arr[:, col_idx]) for tree, col_idx in self._trees],
            axis=1,  # shape: (n_samples, n_estimators)
        )

        if self.task == "classification":
            return self._majority_vote(all_preds)
        return np.mean(all_preds.astype(float), axis=1)

    # ── internals ───────────────────────────────

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.floor(np.sqrt(n_features))))
        if mf == "log2":
            return max(1, int(np.floor(np.log2(n_features))))
        if isinstance(mf, float):
            if not 0.0 < mf <= 1.0:
                raise ValueError("max_features as float must be in (0, 1].")
            return max(1, int(np.floor(mf * n_features)))
        if isinstance(mf, int):
            if not 1 <= mf <= n_features:
                raise ValueError(f"max_features={mf} out of range [1, {n_features}].")
            return mf
        raise ValueError("max_features must be int, float, 'sqrt', 'log2', or None.")

    @staticmethod
    def _majority_vote(all_preds: np.ndarray) -> np.ndarray:
        """Row-wise majority vote (works for any label type)."""
        result = []
        for row in all_preds:
            labels, counts = np.unique(row, return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.array(result)

    def _check_fitted(self) -> None:
        if not self._trees:
            raise RuntimeError("Call fit() before predict().")
