from __future__ import annotations

from typing import Any

import numpy as np

from GlassBox.eda import DataProfiler, IQR_OutlierDetector
from GlassBox.ml import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from GlassBox.numpandas import DataFrame, read_csv
from GlassBox.numpandas.core.series import Series
from GlassBox.numpandas.utils.dtypes import is_nan_value
from GlassBox.optimization import RandomizedSearchCV, cross_val_score
from GlassBox.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

from GlassBox.autofit.detect import detect_task
from GlassBox.autofit.registry import (
    get_default_init_params,
    get_default_search_space,
    get_model,
    get_models_for_task,
)
from GlassBox.autofit.report import build_report


def _missing_count(values: np.ndarray) -> int:
    if np.issubdtype(values.dtype, np.floating):
        return int(np.isnan(values).sum())
    return int(sum(1 for value in values if is_nan_value(value)))


def _build_eda_summary(df: DataFrame) -> dict[str, Any]:
    profiler = DataProfiler(df)
    profiler.compute_profile()

    outlier_detector = IQR_OutlierDetector().fit(df)
    outliers = outlier_detector.get_outlier_report(df)

    correlations: dict[str, float] = {}
    numeric_cols = [
        col
        for col in df.columns
        if np.issubdtype(df.dtypes[col], np.number) and not np.issubdtype(df.dtypes[col], np.bool_)
    ]

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            left = numeric_cols[i]
            right = numeric_cols[j]
            left_arr = df[left].to_numpy().astype(float)
            right_arr = df[right].to_numpy().astype(float)
            valid = (~np.isnan(left_arr)) & (~np.isnan(right_arr))
            if int(valid.sum()) < 2:
                corr = 0.0
            else:
                corr_matrix = np.corrcoef(left_arr[valid], right_arr[valid])
                corr = float(corr_matrix[0, 1]) if np.isfinite(corr_matrix[0, 1]) else 0.0
            correlations[f"{left}_{right}"] = corr

    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "dtypes": {col: str(df.dtypes[col]) for col in df.columns},
        "missing_per_column": {
            col: _missing_count(df[col].to_numpy())
            for col in df.columns
        },
        "outliers_detected": {
            col: int(outliers.get(col, 0))
            for col in df.columns
        },
        "correlations": correlations,
    }


def _merge_frames(left: DataFrame | None, right: DataFrame | None) -> DataFrame:
    data: dict[str, np.ndarray] = {}
    columns: list[str] = []

    for frame in (left, right):
        if frame is None:
            continue
        for col in frame.columns:
            data[col] = frame[col].to_numpy()
            columns.append(col)

    return DataFrame(data, columns=columns)


def _preprocess_features(X: DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    numeric_cols = [
        col
        for col in X.columns
        if np.issubdtype(X.dtypes[col], np.number) and not np.issubdtype(X.dtypes[col], np.bool_)
    ]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    num_out: DataFrame | None = None
    cat_out: DataFrame | None = None

    if numeric_cols:
        X_num = X[numeric_cols]
        num_imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_num = num_imputer.fit_transform(X_num)
        num_out = scaler.fit_transform(X_num)

    if categorical_cols:
        X_cat = X[categorical_cols]
        cat_imputer = SimpleImputer(strategy="most_frequent")
        encoder = OneHotEncoder(handle_unknown="ignore")
        X_cat = cat_imputer.fit_transform(X_cat)
        cat_out = encoder.fit_transform(X_cat)

    transformed = _merge_frames(num_out, cat_out)
    summary = {
        "imputer": "mean+most_frequent" if categorical_cols and numeric_cols else (
            "mean" if numeric_cols else "most_frequent"
        ),
        "scaler": "StandardScaler" if numeric_cols else "",
        "encoders_applied": ["OneHotEncoder"] if categorical_cols else [],
    }
    return transformed, summary


def _instantiate_model(model_name: str):
    model_cls = get_model(model_name)
    init_params = get_default_init_params(model_name)
    return model_cls(**init_params)


def _evaluate(task: str, y_true: Series, y_pred: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
        }

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _train_test_split_frame(
    X: DataFrame,
    y: Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[DataFrame, DataFrame, Series, Series]:
    n_samples = len(X)
    n_test = max(1, int(round(n_samples * test_size)))
    if n_test >= n_samples:
        raise ValueError("test_size leaves no training samples.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_array = X.to_numpy()
    y_array = y.to_numpy()

    X_train_data = {
        col: X_array[train_idx, i].astype(X.dtypes[col])
        for i, col in enumerate(X.columns)
    }
    X_test_data = {
        col: X_array[test_idx, i].astype(X.dtypes[col])
        for i, col in enumerate(X.columns)
    }

    X_train = DataFrame(X_train_data, columns=X.columns)
    X_test = DataFrame(X_test_data, columns=X.columns)
    y_train = Series(y_array[train_idx], name=y.name)
    y_test = Series(y_array[test_idx], name=y.name)

    return X_train, X_test, y_train, y_test


def autofit(
    csv_path: str,
    target_col: str,
    models: list[str] | None = None,
    tuning: bool = True,
) -> dict:
    """Run an end-to-end AutoML workflow and return a JSON-ready report dict."""
    df = read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the DataFrame.")

    task = detect_task(df, target_col)
    eda_summary = _build_eda_summary(df)

    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    X_processed, preprocessing_summary = _preprocess_features(X_raw)

    X_train, X_test, y_train, y_test = _train_test_split_frame(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
    )

    task_models = get_models_for_task(task)

    if models is None:
        selected_models = list(task_models.keys())
    else:
        unknown = [name for name in models if name not in task_models]
        if unknown:
            unknown_csv = ", ".join(unknown)
            raise ValueError(f"Unknown model(s) for task '{task}': {unknown_csv}")
        selected_models = list(models)

    scoring = "accuracy" if task == "classification" else "r2"
    model_results: list[dict[str, Any]] = []

    for model_name in selected_models:
        result: dict[str, Any] = {
            "name": model_name,
            "best_params": {},
            "metrics": {},
            "cv_score": 0.0,
        }

        try:
            estimator = _instantiate_model(model_name)

            if tuning:
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=get_default_search_space(model_name),
                    n_iter=10,
                    cv=5,
                    scoring=scoring,
                    random_state=42,
                )
                search.fit(X_train, y_train)
                fitted_model = search.best_estimator_
                result["best_params"] = dict(search.best_params_ or {})
            else:
                fitted_model = estimator.fit(X_train, y_train)

            y_pred = fitted_model.predict(X_test)
            result["metrics"] = _evaluate(task, y_test, y_pred)

            cv_scores = cross_val_score(
                fitted_model,
                X_train,
                y_train,
                cv=5,
                scoring=scoring,
            )
            result["cv_score"] = float(np.mean(cv_scores)) if cv_scores else 0.0

        except Exception as exc:
            result["error"] = str(exc)

        model_results.append(result)

    return build_report(
        task=task,
        eda_summary=eda_summary,
        preprocessing_summary=preprocessing_summary,
        model_results=model_results,
    )
