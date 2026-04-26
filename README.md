# GlassBox

GlassBox is a lightweight, educational data science toolkit built from scratch with a transparent implementation style. It recreates core ideas from pandas and scikit-learn so users can inspect how dataframes, preprocessing, models, metrics, pipelines, search, and benchmarks work internally.

The project is organized as several small packages under the shared `GlassBox` namespace.

## Packages

| Package | Purpose |
|---|---|
| `glassbox-numpandas` | DataFrame, Series, Index, and basic CSV/JSON/Excel I/O |
| `glassbox-eda` | Statistical profiling, outlier detection, and plotting helpers |
| `glassbox-preprocessing` | Imputers, scalers, encoders, column transformers, SMOTE |
| `glassbox-split` | Train/test and train/validation/test splitting |
| `glassbox-ml` | Linear/logistic regression, KNN, decision tree, random forest, naive Bayes, metrics |
| `glassbox-pipeline` | Sequential preprocessing + estimator workflows |
| `glassbox-optimization` | K-fold cross-validation, grid search, randomized search |
| `glassbox-benchmark` | Comparisons against pandas and scikit-learn |

## Setup

From the repository root:

```powershell
python -m pip install -r requirements-dev.txt
```

If you do not install the packages, run examples from the repository root. The test configuration already adds the package source folders through `pyproject.toml`.

## Quick Example

```python
import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series
from GlassBox.split import train_test_split
from GlassBox.preprocessing import StandardScaler
from GlassBox.pipeline import Pipeline
from GlassBox.ml import LogisticRegressionGD, accuracy_score

X = DataFrame({
    "age": np.array([20, 22, 24, 40, 42, 44]),
    "income": np.array([100, 110, 115, 200, 210, 220]),
})
y = Series(np.array([0, 0, 0, 1, 1, 1]), name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
```

## Hyperparameter Search Example

```python
from GlassBox.optimization import GridSearchCV

param_grid = {
    "clf__learning_rate": [0.01, 0.1, 1.0],
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=2,
    scoring="accuracy",
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)
```

## Benchmarking

The benchmark compares GlassBox with pandas and scikit-learn.

Run:

```powershell
python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py
```

Generated outputs:

```text
packages/glassbox-benchmark/results/comparison_results.csv
packages/glassbox-benchmark/docs/glassbox_comparison_report.md
```

Use custom generated dataset sizes:

```powershell
python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py --rows 10000 --cols 12 --repeats 7
```

The benchmark checks:

- DataFrame `mean` and `sum` against pandas
- `MinMaxScaler` against scikit-learn
- metrics such as `accuracy_score` and `mean_squared_error` against scikit-learn
- model quality and runtime for linear regression, logistic regression, and KNN

## Running Tests

If `pytest` is installed:

```powershell
python -m pytest
```

For a simple end-to-end pipeline check without pytest:

```powershell
python verify.py
```

## Project Goal

GlassBox is not meant to fully replace pandas or scikit-learn. Its goal is to provide a readable, auditable, from-scratch implementation of common data science workflows. The benchmark report should be used honestly: GlassBox can match many outputs and is fast for some simple operations, while mature optimized libraries are still faster for many model-training tasks.
