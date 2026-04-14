import unittest
import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from GlassBox.preprocessing.impute import SimpleImputer
from GlassBox.preprocessing.scale import StandardScaler, MinMaxScaler, RobustScaler
from GlassBox.preprocessing.encode import OneHotEncoder, OrdinalEncoder, LabelEncoder
from GlassBox.preprocessing.compose import ColumnTransformer, make_column_transformer
from GlassBox.preprocessing.smote import SMOTE

class TestImputer(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            "a": [1, 2, np.nan, 4],
            "b": np.array([np.nan, "cat", "dog", "cat"], dtype=object)
        })

    def test_impute_mean(self):
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(self.df[["a"]])
        res = imputer.transform(self.df[["a"]])
        self.assertAlmostEqual(res["a"].to_numpy()[2], (1+2+4)/3)

    def test_impute_most_frequent(self):
        imputer = SimpleImputer(strategy="most_frequent")
        res = imputer.fit_transform(self.df)
        self.assertEqual(res["b"].to_numpy()[0], "cat")

    def test_imputation_indicator(self):
        imputer = SimpleImputer(strategy="mean", imputation_indicator=True)
        res = imputer.fit_transform(self.df[["a"]])
        self.assertEqual(res.columns[1], "a_imputed")
        self.assertEqual(res["a_imputed"].to_numpy()[2], 1)
        self.assertEqual(res["a_imputed"].to_numpy()[0], 0)

class TestScalers(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })

    def test_standard_scaler(self):
        scaler = StandardScaler()
        res = scaler.fit_transform(self.df)
        # mean is 3, std is approx 1.58
        self.assertAlmostEqual(res["a"].to_numpy().mean(), 0)
        self.assertAlmostEqual(res["a"].to_numpy().std(ddof=1), 1)

    def test_minmax_scaler(self):
        scaler = MinMaxScaler()
        res = scaler.fit_transform(self.df)
        self.assertEqual(res["a"].to_numpy().min(), 0)
        self.assertEqual(res["a"].to_numpy().max(), 1)

class TestEncoders(unittest.TestCase):
    def setUp(self):
        self.df = DataFrame({
            "color": ["red", "blue", "red", "green"]
        })
        self.target = Series(["cat", "dog", "cat"])

    def test_onehot_encoder(self):
        enc = OneHotEncoder(handle_unknown="ignore")
        res = enc.fit_transform(self.df)
        self.assertIn("color_red", res.columns)
        self.assertIn("color_blue", res.columns)
        self.assertEqual(res["color_red"].to_numpy()[0], 1)

    def test_ordinal_encoder(self):
        enc = OrdinalEncoder()
        res = enc.fit_transform(self.df)
        arr = res["color"].to_numpy()
        self.assertTrue(np.all(arr >= 0))
        self.assertEqual(len(np.unique(arr)), 3)

    def test_label_encoder(self):
        enc = LabelEncoder()
        res = enc.fit_transform(self.target)
        self.assertEqual(len(res), 3)
        self.assertTrue(np.max(res) <= 1)
        
class TestSMOTE(unittest.TestCase):
    def setUp(self):
        # Create an imbalanced dataset with numerical and categorical features
        self.X = DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5],
            "cat": ["a", "b", "a", "c", "b", "a", "a"]
        })
        self.y = Series([0, 0, 0, 0, 0, 1, 1], name="target")
        
    def test_smote_resample(self):
        smote = SMOTE(k_neighbors=1, random_state=42)
        X_res, y_res = smote.fit_resample(self.X, self.y)
        
        # Verify sizes (minority class 1 should be upsampled to 5)
        self.assertEqual(len(y_res), 10)
        self.assertEqual(np.sum(y_res.to_numpy() == 1), 5)
        self.assertEqual(np.sum(y_res.to_numpy() == 0), 5)
        
        # Verify columns exist and are same
        self.assertListEqual(X_res.columns, ["num", "cat"])
        # Verify categorical interpolation preserved valid types
        cat_vals = set(X_res["cat"].to_numpy())
        self.assertTrue(cat_vals.issubset({"a", "b", "c"}))

if __name__ == "__main__":
    unittest.main()
