import unittest
import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series
from GlassBox.split import train_test_split, train_validation_test_split

class TestSplit(unittest.TestCase):
    def setUp(self):
        self.X = DataFrame({
            "a": np.arange(100),
            "b": np.arange(100) * 2
        })
        self.y = Series(np.arange(100), name="target")

    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Verify columns preserved
        self.assertListEqual(X_train.columns, ["a", "b"])
        # Verify alignment
        idx = X_train["a"].to_numpy()[0]
        self.assertEqual(y_train.to_numpy()[0], idx)

    def test_train_validation_test_split(self):
        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
            self.X, self.y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42
        )
        
        self.assertEqual(len(X_train), 60)
        self.assertEqual(len(X_val), 20)
        self.assertEqual(len(X_test), 20)
        
        self.assertEqual(len(y_train), 60)
        self.assertEqual(len(y_val), 20)
        self.assertEqual(len(y_test), 20)

if __name__ == "__main__":
    unittest.main()
