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
        # y is heavily imbalanced: 90 zeros, 10 ones
        self.y = Series(np.concatenate([np.zeros(90), np.ones(10)]), name="target")

    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Verify columns preserved
        self.assertListEqual(X_train.columns, ["a", "b"])
        # Verify alignment
        idx = int(X_train["a"].to_numpy()[0])
        self.assertEqual(y_train.to_numpy()[0], self.y.to_numpy()[idx])

    def test_train_test_split_stratify(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        
        # For an 80/20 split on 90/10 ratio:
        # test should have 20 elements, where 10% are ones (so 2 ones, 18 zeros)
        self.assertEqual(np.sum(y_test.to_numpy()), 2)
        self.assertEqual(np.sum(y_train.to_numpy()), 8)

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

    def test_train_validation_test_split_stratify(self):
        # Using a dataset with simple numbers easily divisible
        X = DataFrame({"a": np.arange(100)})
        y = Series(np.concatenate([np.zeros(50), np.ones(50)]), name="target")
        
        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
            X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test 20 elements => 10 ones, 10 zeros
        self.assertEqual(np.sum(y_test.to_numpy() == 1), 10)
        # Val 20 elements => 10 ones, 10 zeros
        self.assertEqual(np.sum(y_val.to_numpy() == 1), 10)
        # Train 60 elements => 30 ones, 30 zeros
        self.assertEqual(np.sum(y_train.to_numpy() == 1), 30)

if __name__ == "__main__":
    unittest.main()
