"""Targeted tests for confusion matrix and classification report utilities.

Run:
    python -m pytest packages/glassbox-eval/tests/test_eval.py
"""

from __future__ import annotations

import unittest

import numpy as np

from GlassBox.eval import binary_confusion_counts, classification_report, confusion_matrix


class TestEvalMetrics(unittest.TestCase):
    def test_confusion_matrix_multiclass(self):
        y_true = ["cat", "dog", "cat", "bird", "dog", "cat"]
        y_pred = ["cat", "cat", "cat", "bird", "dog", "dog"]
        labels = ["bird", "cat", "dog"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        expected = np.array(
            [
                [1, 0, 0],
                [0, 2, 1],
                [0, 1, 1],
            ]
        )
        np.testing.assert_array_equal(cm, expected)

    def test_binary_confusion_counts(self):
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 0, 1, 0, 1, 0]

        counts = binary_confusion_counts(y_true, y_pred, positive_label=1)

        self.assertEqual(counts["true_positives"], 2)
        self.assertEqual(counts["false_negatives"], 1)
        self.assertEqual(counts["false_positives"], 1)
        self.assertEqual(counts["true_negatives"], 2)

    def test_classification_report_output_dict(self):
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 0, 1, 0, 0]

        report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True)

        self.assertIn("0", report)
        self.assertIn("1", report)
        self.assertIn("accuracy", report)
        self.assertIn("macro avg", report)
        self.assertIn("weighted avg", report)

        self.assertAlmostEqual(report["accuracy"], 0.8)
        self.assertAlmostEqual(report["1"]["precision"], 1.0)
        self.assertAlmostEqual(report["1"]["recall"], 2 / 3)

    def test_classification_report_string(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        report_str = classification_report(y_true, y_pred, output_dict=False, digits=3)

        self.assertIn("precision", report_str)
        self.assertIn("recall", report_str)
        self.assertIn("f1-score", report_str)
        self.assertIn("accuracy", report_str)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            confusion_matrix([1, 0, 1], [1, 0])


if __name__ == "__main__":
    unittest.main()
