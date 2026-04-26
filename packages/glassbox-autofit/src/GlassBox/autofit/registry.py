from __future__ import annotations

from GlassBox.ml import (
    DecisionTree,
    GaussianNaiveBayes,
    KNNClassifier,
    KNNRegressor,
    LinearRegressionGD,
    LogisticRegressionGD,
    RandomForest,
)


_CLASSIFICATION_MODELS: dict[str, type] = {
    "LogisticRegression": LogisticRegressionGD,
    "KNNClassifier": KNNClassifier,
    "DecisionTreeClassifier": DecisionTree,
    "RandomForestClassifier": RandomForest,
    "NaiveBayes": GaussianNaiveBayes,
}

_REGRESSION_MODELS: dict[str, type] = {
    "LinearRegression": LinearRegressionGD,
    "KNNRegressor": KNNRegressor,
    "DecisionTreeRegressor": DecisionTree,
    "RandomForestRegressor": RandomForest,
}

_DEFAULT_INIT_PARAMS: dict[str, dict] = {
    "LogisticRegression": {},
    "KNNClassifier": {},
    "DecisionTreeClassifier": {"task": "classification"},
    "RandomForestClassifier": {"task": "classification"},
    "NaiveBayes": {},
    "LinearRegression": {},
    "KNNRegressor": {},
    "DecisionTreeRegressor": {"task": "regression"},
    "RandomForestRegressor": {"task": "regression"},
}

_DEFAULT_SEARCH_SPACE: dict[str, dict[str, list]] = {
    "LogisticRegression": {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_iterations": [500, 1000, 2000],
    },
    "KNNClassifier": {
        "n_neighbors": [3, 5, 7],
        "distance_metric": ["euclidean", "manhattan"],
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 4, 8],
    },
    "RandomForestClassifier": {
        "n_estimators": [25, 50, 100],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "NaiveBayes": {
        "var_smoothing": [1e-9, 1e-8, 1e-7],
    },
    "LinearRegression": {
        "learning_rate": [0.005, 0.01, 0.05],
        "n_iterations": [500, 1000, 2000],
    },
    "KNNRegressor": {
        "n_neighbors": [3, 5, 7],
        "distance_metric": ["euclidean", "manhattan"],
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 4, 8],
    },
    "RandomForestRegressor": {
        "n_estimators": [25, 50, 100],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
}


def get_models_for_task(task: str) -> dict[str, type]:
    """Return model mapping for the provided task."""
    if task == "classification":
        return dict(_CLASSIFICATION_MODELS)
    if task == "regression":
        return dict(_REGRESSION_MODELS)
    raise ValueError("task must be either 'classification' or 'regression'.")


def get_model(name: str) -> type:
    """Return a model class by name."""
    all_models = {**_CLASSIFICATION_MODELS, **_REGRESSION_MODELS}
    if name not in all_models:
        raise ValueError(f"Unknown model '{name}'.")
    return all_models[name]


def get_default_init_params(name: str) -> dict:
    """Return default constructor keyword arguments for a model name."""
    if name not in _DEFAULT_INIT_PARAMS:
        raise ValueError(f"Unknown model '{name}'.")
    return dict(_DEFAULT_INIT_PARAMS[name])


def get_default_search_space(name: str) -> dict[str, list]:
    """Return default RandomizedSearchCV space for a model name."""
    if name not in _DEFAULT_SEARCH_SPACE:
        raise ValueError(f"Unknown model '{name}'.")
    return {key: list(values) for key, values in _DEFAULT_SEARCH_SPACE[name].items()}
