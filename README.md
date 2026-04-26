# GlassBox

GlassBox is a lightweight educational data science toolkit built from scratch. It recreates important parts of pandas and scikit-learn in a transparent way, so the code can be inspected and explained.

The easiest way to understand and use the project is to run `verify.py`.

## Setup

From the repository root:

```powershell
python -m pip install -r requirements-dev.txt
```

If you do not install the packages, run examples from the repository root. The test configuration already adds the package source folders through `pyproject.toml`.

## Main Usage Example

Run the included end-to-end example:

```powershell
python verify.py
```

This script demonstrates the main flow of the project:

1. Add all local package source folders to Python path.
2. Create a small synthetic dataset with `DataFrame` and `Series`.
3. Split the data into training and testing sets.
4. Build a machine learning pipeline.
5. Apply preprocessing with `StandardScaler`.
6. Balance the imbalanced dataset with `SMOTE`.
7. Train `LogisticRegressionGD`.
8. Tune the learning rate with `GridSearchCV`.
9. Predict on the test data.
10. Print the best parameters, best score, and prediction count.

Expected output looks like:

```text
Testing pipeline end-to-end...
Best params: {'clf__learning_rate': ...}
Best score: ...
Predictions len: ...
Test finished successfully!
```

The exact score may change because `verify.py` generates random data.

## What `verify.py` Uses

The example imports and connects the main GlassBox modules:

```python
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series
from GlassBox.split import train_test_split
from GlassBox.preprocessing import SMOTE, StandardScaler
from GlassBox.pipeline import Pipeline
from GlassBox.optimization import GridSearchCV
from GlassBox.ml.linear_model import LogisticRegressionGD
```



## Project Structure

| Package | Role |
|---|---|
| `glassbox-numpandas` | Provides `DataFrame` and `Series` |
| `glassbox-preprocessing` | Provides scalers, imputers, encoders, and SMOTE |
| `glassbox-split` | Splits data into train/test sets |
| `glassbox-pipeline` | Chains preprocessing and models |
| `glassbox-optimization` | Provides `GridSearchCV` and cross-validation |
| `glassbox-ml` | Provides models and metrics |
| `glassbox-eda` | Provides profiling and exploration helpers |
| `glassbox-benchmark` | Compares GlassBox with pandas/scikit-learn |

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

## Tests

If `pytest` is installed:

```powershell
python -m pytest
```

Without pytest, use the main verification script:

```powershell
python verify.py
```

## Project Goal

GlassBox is not meant to fully replace pandas or scikit-learn. Its goal is to provide a readable, auditable, from-scratch implementation of common data science workflows. The benchmark report should be used honestly: GlassBox can match many outputs and is fast for some simple operations, while mature optimized libraries are still faster for many model-training tasks.
