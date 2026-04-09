"""Targeted tests for the minimal ML module.

Run:
    python -m unittest packages/glassbox-ml/tests/test_ml.py
"""

from __future__ import annotations

import numpy as np

from GlassBox.ml import (
    DecisionTree,
    GaussianNaiveBayes,
    KNNClassifier,
    KNNRegressor,
    LinearRegressionGD,
    LogisticRegressionGD,
    RandomForest,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    train_test_split,
)


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
    assert_close(mean_absolute_error(y_true, y_pred), 0.5)
    assert_close(mean_squared_error(y_true, y_pred), 0.375)
    assert_close(r2_score(y_true, y_pred), 0.9486081370449679)


def test_classification_metrics():
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]
    assert_close(accuracy_score(y_true, y_pred), 0.8)
    assert_close(precision_score(y_true, y_pred), 1.0)
    assert_close(recall_score(y_true, y_pred), 2 / 3)
    assert_close(f1_score(y_true, y_pred), 0.8)


def test_linear_regression_gd():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)
    model = LinearRegressionGD(learning_rate=0.05, n_iterations=5000)
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
    model = LogisticRegressionGD(learning_rate=0.2, n_iterations=4000)
    model.fit(X, y)
    preds = model.predict(X)
    assert accuracy_score(y, preds) >= 5 / 6
    probas = model.predict_proba([[2.0, 2.0]])
    assert probas.shape == (1, 2)
    assert probas[0, 1] > 0.5


def test_knn_classifier_and_regressor():
    X_class = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6], [6, 5]]
    y_class = [0, 0, 0, 1, 1, 1]
    clf = KNNClassifier(n_neighbors=3, distance_metric="euclidean")
    clf.fit(X_class, y_class)
    assert clf.predict([[0.2, 0.1], [5.4, 5.1]]).tolist() == [0, 1]

    X_reg = [[1], [2], [3], [4], [5]]
    y_reg = [2, 4, 6, 8, 10]
    reg = KNNRegressor(n_neighbors=2, distance_metric="manhattan")
    reg.fit(X_reg, y_reg)
    preds = reg.predict([[1.5], [4.5]])
    assert np.allclose(preds, [3.0, 9.0])


def test_train_test_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )
    assert X_train.shape == (7, 2)
    assert X_test.shape == (3, 2)
    assert y_train.shape == (7,)
    assert y_test.shape == (3,)


def test_decision_tree_classifier_and_regressor():
    X_class = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y_class = np.array(["no", "no", "yes", "yes"], dtype=object)
    clf = DecisionTree(task="classification", max_depth=2)
    clf.fit(X_class, y_class)
    assert clf.predict([[0.0, 0.0], [1.0, 1.0]]).tolist() == ["no", "yes"]

    X_reg = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y_reg = np.array([2, 4, 6, 8, 10], dtype=float)
    reg = DecisionTree(task="regression", max_depth=3)
    reg.fit(X_reg, y_reg)
    preds = reg.predict([[1], [5]]).astype(float)
    assert np.allclose(preds, [2.0, 10.0], atol=1e-6)


def test_gaussian_naive_bayes_predict_and_proba():
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
    y = np.array(["cold", "cold", "cold", "hot", "hot", "hot"], dtype=object)
    model = GaussianNaiveBayes()
    model.fit(X, y)

    preds = model.predict([[0.0, 0.0], [2.0, 2.0]])
    assert preds.tolist() == ["cold", "hot"]

    proba = model.predict_proba([[0.0, 0.0], [2.0, 2.0]])
    assert proba.shape == (2, 2)
    assert np.allclose(np.sum(proba, axis=1), [1.0, 1.0], atol=1e-9)


def test_random_forest_classifier_and_regressor():
    X_class = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
    y_class = np.array(["cold", "cold", "cold", "hot", "hot", "hot"], dtype=object)
    clf = RandomForest(
        task="classification",
        n_estimators=15,
        max_depth=4,
        random_state=11,
    )
    clf.fit(X_class, y_class)
    class_preds = clf.predict([[0.0, 0.0], [2.0, 2.0]])
    assert class_preds.tolist() == ["cold", "hot"]

    X_reg = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
    y_reg = np.array([2, 4, 6, 8, 10, 12], dtype=float)
    reg = RandomForest(
        task="regression",
        n_estimators=21,
        max_depth=5,
        random_state=19,
    )
    reg.fit(X_reg, y_reg)
    reg_preds = reg.predict([[1.5], [5.5]]).astype(float)
    assert reg_preds.shape == (2,)
    assert np.all(np.isfinite(reg_preds))
    assert reg_preds[1] > reg_preds[0]
    assert np.all(reg_preds >= y_reg.min())
    assert np.all(reg_preds <= y_reg.max())


def main():
    tests = [
        ("regression metrics", test_regression_metrics),
        ("classification metrics", test_classification_metrics),
        ("linear regression GD", test_linear_regression_gd),
        ("logistic regression GD", test_logistic_regression_gd),
        ("KNN classifier/regressor", test_knn_classifier_and_regressor),
        ("train_test_split", test_train_test_split),
        (
            "decision tree classifier/regressor",
            test_decision_tree_classifier_and_regressor,
        ),
        ("gaussian naive bayes", test_gaussian_naive_bayes_predict_and_proba),
        (
            "random forest classifier/regressor",
            test_random_forest_classifier_and_regressor,
        ),
    ]
    for name, fn in tests:
        run_test(name, fn)


if __name__ == "__main__":
    main()
