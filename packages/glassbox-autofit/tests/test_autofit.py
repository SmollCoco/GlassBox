import csv
import os
import tempfile
import unittest

import numpy as np

from GlassBox.autofit import autofit
from GlassBox.pipeline import Pipeline


class TestAutoFit(unittest.TestCase):
    def _write_temp_csv(self, headers, rows):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        with handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        return handle.name

    def _classification_csv(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 3))
        y = (X[:, 0] + 0.7 * X[:, 1] > 0).astype(int)
        rows = [
            [float(X[i, 0]), float(X[i, 1]), float(X[i, 2]), int(y[i])]
            for i in range(100)
        ]
        return self._write_temp_csv(["f1", "f2", "f3", "target"], rows)

    def _regression_csv(self):
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, size=(100, 3))
        noise = rng.normal(0, 0.1, size=100)
        y = 2.5 * X[:, 0] - 1.2 * X[:, 1] + 0.8 * X[:, 2] + noise
        rows = [
            [float(X[i, 0]), float(X[i, 1]), float(X[i, 2]), float(y[i])]
            for i in range(100)
        ]
        return self._write_temp_csv(["f1", "f2", "f3", "target"], rows)

    def _assert_report_shape(self, report):
        self.assertIn("task", report)
        self.assertIn("eda", report)
        self.assertIn("preprocessing", report)
        self.assertIn("models", report)
        self.assertIn("best_model", report)

        self.assertIn("shape", report["eda"])
        self.assertIn("dtypes", report["eda"])
        self.assertIn("missing_per_column", report["eda"])
        self.assertIn("outliers_detected", report["eda"])
        self.assertIn("correlations", report["eda"])

        self.assertIn("imputer", report["preprocessing"])
        self.assertIn("scaler", report["preprocessing"])
        self.assertIn("encoders_applied", report["preprocessing"])

        self.assertIsInstance(report["models"], list)
        self.assertGreater(len(report["models"]), 0)

    def test_autofit_classification_models_none_tuning_false(self):
        csv_path = self._classification_csv()
        try:
            report, fitted_pipeline = autofit(
                csv_path, target_col="target", models=None, tuning=False
            )
            self._assert_report_shape(report)
            self.assertEqual(report["task"], "classification")
            self.assertIsInstance(fitted_pipeline, Pipeline)
        finally:
            os.remove(csv_path)

    def test_autofit_regression_models_none_tuning_false(self):
        csv_path = self._regression_csv()
        try:
            report, fitted_pipeline = autofit(
                csv_path, target_col="target", models=None, tuning=False
            )
            self._assert_report_shape(report)
            self.assertEqual(report["task"], "regression")
            self.assertIsInstance(fitted_pipeline, Pipeline)
        finally:
            os.remove(csv_path)

    def test_autofit_specific_model_list(self):
        csv_path = self._classification_csv()
        try:
            report, fitted_pipeline = autofit(
                csv_path,
                target_col="target",
                models=["KNNClassifier", "LogisticRegression"],
                tuning=False,
            )
            names = [entry.get("name") for entry in report["models"]]
            self.assertListEqual(names, ["KNNClassifier", "LogisticRegression"])
            self.assertIsInstance(fitted_pipeline, Pipeline)
        finally:
            os.remove(csv_path)

    def test_autofit_unknown_model_raises(self):
        csv_path = self._classification_csv()
        try:
            with self.assertRaises(ValueError):
                autofit(
                    csv_path, target_col="target", models=["UnknownModel"], tuning=False
                )
        finally:
            os.remove(csv_path)

    def test_autofit_missing_target_raises(self):
        csv_path = self._classification_csv()
        try:
            with self.assertRaises(ValueError):
                autofit(csv_path, target_col="does_not_exist", tuning=False)
        finally:
            os.remove(csv_path)


if __name__ == "__main__":
    unittest.main()
