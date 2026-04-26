import unittest
import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from GlassBox.optimization import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV

class DummyEstimator:
    def __init__(self, param1=1):
        self.param1 = param1
        self.fitted = False
        
    def fit(self, X, y):
        self.fitted = True
        return self
        
    def predict(self, X):
        return np.zeros(len(X))

class TestOptimization(unittest.TestCase):
    def setUp(self):
        self.X = DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.y = Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="target")

    def test_kfold(self):
        kf = KFold(n_splits=2)
        splits = list(kf.split(self.X))
        self.assertEqual(len(splits), 2)
        
        train_idx, test_idx = splits[0]
        self.assertEqual(len(train_idx), 5)
        self.assertEqual(len(test_idx), 5)

    def test_cross_val_score(self):
        est = DummyEstimator()
        scores = cross_val_score(est, self.X, self.y, cv=2, scoring="accuracy")
        self.assertEqual(len(scores), 2)
        # the dummy predicts all 0s, and true y test is [0, 1, 0, 1, 0]. Accuracy is 3/5 = 0.6
        self.assertAlmostEqual(scores[0], 0.6)

    def test_grid_search(self):
        est = DummyEstimator()
        param_grid = {"param1": [1, 2, 3]}
        grid = GridSearchCV(est, param_grid, cv=2, scoring="accuracy")
        grid.fit(self.X, self.y)
        
        self.assertIsNotNone(grid.best_params_)
        self.assertEqual(len(grid.cv_results_), 3)

    def test_randomized_search(self):
        est = DummyEstimator()
        param_dist = {"param1": [1, 2, 3, 4, 5]}
        search = RandomizedSearchCV(est, param_dist, n_iter=2, cv=2, scoring="accuracy", random_state=42)
        search.fit(self.X, self.y)
        
        self.assertIsNotNone(search.best_params_)
        self.assertEqual(len(search.cv_results_), 2)

if __name__ == "__main__":
    unittest.main()
