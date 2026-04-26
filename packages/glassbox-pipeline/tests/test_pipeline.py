import unittest
import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from GlassBox.pipeline import Pipeline

class DummyTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

class DummyResampler:
    def fit_resample(self, X, y):
        # Double the data
        X_new = DataFrame({col: np.concatenate([X[col].to_numpy(), X[col].to_numpy()]) for col in X.columns})
        y_new = Series(np.concatenate([y.to_numpy(), y.to_numpy()]))
        return X_new, y_new

class DummyEstimator:
    def __init__(self):
        self.fitted = False
    def fit(self, X, y=None):
        self.fitted = True
        return self
    def predict(self, X):
        return np.ones(len(X))

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.X = DataFrame({"a": [1, 2, 3]})
        self.y = Series([0, 1, 0])

    def test_pipeline_fit_predict(self):
        pipe = Pipeline([
            ("t1", DummyTransformer()),
            ("clf", DummyEstimator())
        ])
        
        pipe.fit(self.X, self.y)
        self.assertTrue(pipe.steps[-1][1].fitted)
        
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), 3)

    def test_pipeline_with_resampler(self):
        pipe = Pipeline([
            ("resample", DummyResampler()),
            ("t1", DummyTransformer()),
            ("clf", DummyEstimator())
        ])
        
        pipe.fit(self.X, self.y)
        self.assertTrue(pipe.steps[-1][1].fitted)
        
        # predict shouldn't use resampler, so output size is 3
        preds = pipe.predict(self.X)
        self.assertEqual(len(preds), 3)

if __name__ == "__main__":
    unittest.main()
