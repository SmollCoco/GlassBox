"""Targeted tests for the minimal ML module.

Run:
    python numpandas/tests/ml_test.py
"""

from __future__ import annotations

import numpy as np

import numpandas as npd


def run_test(name, fn):
    try:
        fn()
        print(f"PASS - {name}")
    except Exception as exc:
        print(f"FAIL - {name}: {type(exc).__name__}: {exc}")


def assert_close(actual, expected, tol=1e-6):
    if not np.isclose(actual, expected, atol=tol):
        raise AssertionError(f"Expected {expected}, got {actual}")


def test_regression_metrics():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]
    assert_close(npd.mean_absolute_error(y_true, y_pred), 0.5)
    assert_close(npd.mean_squared_error(y_true, y_pred), 0.375)
    assert_close(npd.r2_score(y_true, y_pred), 0.9486081370449679)


def test_classification_metrics():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    assert_close(npd.accuracy_score(y_true, y_pred), 0.8)
    assert_close(npd.precision_score(y_true, y_pred), 1.0)
    assert_close(npd.recall_score(y_true, y_pred), 2 / 3)
    assert_close(npd.f1_score(y_true, y_pred), 0.8)


def test_linear_regression_gd():
    X = npd.DataFrame({"x": [1, 2, 3, 4, 5]})
    y = npd.Series([3, 5, 7, 9, 11])
    model = npd.LinearRegressionGD(learning_rate=0.05, n_iterations=5000)
    model.fit(X, y)
    predictions = model.predict([[6], [7]])
    assert np.allclose(predictions, [13, 15], atol=0.2)
    assert model.score(X, y) > 0.999


def test_logistic_regression_gd():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    model = npd.LogisticRegressionGD(learning_rate=0.2, n_iterations=4000)
    model.fit(X, y)
    preds = model.predict(X)
    assert npd.accuracy_score(y, preds) >= 5 / 6
    probas = model.predict_proba([[2.0, 2.0]])
    assert probas.shape == (1, 2)
    assert probas[0, 1] > 0.5


def test_knn_classifier_and_regressor():
    X_class = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]]
    y_class = [0, 0, 0, 1, 1, 1]
    clf = npd.KNNClassifier(n_neighbors=3, distance_metric="euclidean")
    clf.fit(X_class, y_class)
    assert clf.predict([[0.2, 0.1], [5.4, 5.1]]).tolist() == [0, 1]

    X_reg = [[1], [2], [3], [4], [5]]
    y_reg = [2, 4, 6, 8, 10]
    reg = npd.KNNRegressor(n_neighbors=2, distance_metric="manhattan")
    reg.fit(X_reg, y_reg)
    preds = reg.predict([[1.5], [4.5]])
    assert np.allclose(preds, [3.0, 9.0])


def test_train_test_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = npd.train_test_split(X, y, test_size=0.3, random_state=7)
    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def main():
    tests = [
        ("regression metrics", test_regression_metrics),
        ("classification metrics", test_classification_metrics),
        ("linear regression GD", test_linear_regression_gd),
        ("logistic regression GD", test_logistic_regression_gd),
        ("KNN classifier/regressor", test_knn_classifier_and_regressor),
        ("train_test_split", test_train_test_split),
    ]
    for name, fn in tests:
        run_test(name, fn)


if __name__ == "__main__":
    main()
