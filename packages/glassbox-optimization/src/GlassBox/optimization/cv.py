import copy
import numpy as np
from typing import Any, Callable

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series


class KFold:
    """K-Folds cross-validator."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate indices to split data into training and test set."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
            
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop


def _get_metric(scoring: str) -> Callable:
    import GlassBox.ml.metrics as metrics
    
    scoring_map = {
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
        "f1": metrics.f1_score,
        "mae": metrics.mean_absolute_error,
        "mse": metrics.mean_squared_error,
        "r2": metrics.r2_score,
    }
    
    if scoring not in scoring_map:
        raise ValueError(f"Scoring metric '{scoring}' not found. Available: {list(scoring_map.keys())}")
        
    return scoring_map[scoring]


def cross_val_score(estimator: Any, X: DataFrame, y: Series, cv: int | KFold = 5, scoring: str = "accuracy") -> list[float]:
    """Evaluate a score by cross-validation."""
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
        
    metric_func = _get_metric(scoring)
    scores = []
    
    for train_idx, test_idx in cv.split(X):
        # Slice X
        X_arr = np.column_stack([X[col].to_numpy() for col in X.columns])
        X_train_arr = X_arr[train_idx]
        X_test_arr = X_arr[test_idx]
        
        train_data = {col: X_train_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
        test_data = {col: X_test_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
        
        X_train = DataFrame(train_data, columns=X.columns)
        X_test = DataFrame(test_data, columns=X.columns)
        
        # Slice y
        y_train = Series(y.to_numpy()[train_idx], name=y.name)
        y_test = Series(y.to_numpy()[test_idx], name=y.name)
        
        # Clone estimator
        cloned_estimator = copy.deepcopy(estimator)
        
        # Train and predict
        cloned_estimator.fit(X_train, y_train)
        y_pred = cloned_estimator.predict(X_test)
        
        # Score
        score = metric_func(y_test, y_pred)
        scores.append(score)
        
    return scores
