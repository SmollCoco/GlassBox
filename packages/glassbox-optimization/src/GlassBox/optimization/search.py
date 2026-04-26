import copy
import itertools
import numpy as np
from typing import Any

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series
from .cv import KFold, cross_val_score


class GridSearchCV:
    """Exhaustive search over specified parameter values for an estimator."""
    
    def __init__(self, estimator: Any, param_grid: dict[str, list], cv: int | KFold = 5, scoring: str = "accuracy"):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = []

    def fit(self, X: DataFrame, y: Series) -> "GridSearchCV":
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        
        best_score = -np.inf if self.scoring in ("accuracy", "precision", "recall", "f1", "r2") else np.inf
        # For error metrics, lower is better. Let's align this:
        is_error_metric = self.scoring in ("mae", "mse")
        
        best_params = None
        
        for combo in combinations:
            params = dict(zip(keys, combo))
            
            cloned = copy.deepcopy(self.estimator)
            # Set params directly and handle pipelines if necessary
            for k, v in params.items():
                if hasattr(cloned, 'steps') and '__' in k: # Handle Pipeline (e.g. step_name__param)
                    step_name, param_name = k.split('__', 1)
                    for step_tpl in cloned.steps:
                        if step_tpl[0] == step_name:
                            setattr(step_tpl[1], param_name, v)
                            break
                else:
                    setattr(cloned, k, v)
                
            scores = cross_val_score(cloned, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = float(np.mean(scores))
            
            self.cv_results_.append({"params": params, "mean_score": mean_score, "scores": scores})
            
            if is_error_metric:
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
            else:
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
        self.best_score_ = best_score
        self.best_params_ = best_params
        
        # Fit final estimator on all data
        self.best_estimator_ = copy.deepcopy(self.estimator)
        for k, v in self.best_params_.items():
            if hasattr(self.best_estimator_, 'steps') and '__' in k:
                step_name, param_name = k.split('__', 1)
                for step_tpl in self.best_estimator_.steps:
                    if step_tpl[0] == step_name:
                        setattr(step_tpl[1], param_name, v)
                        break
            else:
                setattr(self.best_estimator_, k, v)
                
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X: DataFrame):
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before predict.")
        return self.best_estimator_.predict(X)


class RandomizedSearchCV:
    """Randomized search on hyper parameters for an estimator."""
    
    def __init__(self, estimator: Any, param_distributions: dict[str, list], n_iter: int = 10, cv: int | KFold = 5, scoring: str = "accuracy", random_state: int = None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = []

    def fit(self, X: DataFrame, y: Series) -> "RandomizedSearchCV":
        rng = np.random.RandomState(self.random_state)
        
        is_error_metric = self.scoring in ("mae", "mse")
        best_score = float('inf') if is_error_metric else float('-inf')
        best_params = None
        
        for _ in range(self.n_iter):
            params = {}
            for k, v in self.param_distributions.items():
                # Choice from list distribution
                params[k] = v[rng.randint(len(v))]
                
            cloned = copy.deepcopy(self.estimator)
            for k, v in params.items():
                if hasattr(cloned, 'steps') and '__' in k:
                    step_name, param_name = k.split('__', 1)
                    for step_tpl in cloned.steps:
                        if step_tpl[0] == step_name:
                            setattr(step_tpl[1], param_name, v)
                            break
                else:
                    setattr(cloned, k, v)
                    
            scores = cross_val_score(cloned, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = float(np.mean(scores))
            
            self.cv_results_.append({"params": params, "mean_score": mean_score, "scores": scores})
            
            if is_error_metric:
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = params
            else:
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
                    
        self.best_score_ = best_score
        self.best_params_ = best_params
        
        # Fit final estimator on all data
        self.best_estimator_ = copy.deepcopy(self.estimator)
        for k, v in self.best_params_.items():
            if hasattr(self.best_estimator_, 'steps') and '__' in k:
                step_name, param_name = k.split('__', 1)
                for step_tpl in self.best_estimator_.steps:
                    if step_tpl[0] == step_name:
                        setattr(step_tpl[1], param_name, v)
                        break
            else:
                setattr(self.best_estimator_, k, v)
                
        self.best_estimator_.fit(X, y)
        return self
        
    def predict(self, X: DataFrame):
        if self.best_estimator_ is None:
            raise ValueError("RandomizedSearchCV must be fitted before predict.")
        return self.best_estimator_.predict(X)
