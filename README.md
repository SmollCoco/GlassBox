# GlassBox

GlassBox is a transparent, educational machine-learning ecosystem implemented as a Python multi-package monorepo. It provides pandas-like tabular data structures, exploratory data analysis, preprocessing, classical machine-learning models, dataset splitting, pipeline orchestration, hyperparameter search, benchmarking, and an AutoFit workflow.

The project is intentionally "glass box": the main data and ML behavior is implemented in readable Python and NumPy-oriented code so users can inspect how the pieces work instead of treating them as black-box wrappers.

## Project Architecture

GlassBox uses a shared Python namespace across independently installable packages:

```text
GlassBox.numpandas       Core DataFrame, Series, Index, and I/O
GlassBox.eda             Profiling, statistics, and plotting
GlassBox.preprocessing   Imputation, scaling, encoding, SMOTE, composition
GlassBox.ml              Models, metrics, and ML utilities
GlassBox.split           Train/test and train/validation/test splitting
GlassBox.pipeline        Sequential transformer/model pipelines
GlassBox.optimization    Cross-validation and hyperparameter search
GlassBox.benchmark       Local comparison benchmarks against pandas/sklearn
GlassBox.autofit         End-to-end AutoML orchestration
```

The repository also includes:

- `packages/glassbox-meta`: umbrella package that depends on the core GlassBox packages.
- `agent/`: NemoClaw/OpenClaw integration for running AutoFit as an agent tool.
- `Dockerfile`: containerized AutoFit runtime with explicit `PYTHONPATH` wiring for the monorepo.
- `benchmark_outputs/` and `packages/glassbox-benchmark/docs/`: benchmark reports.

## Modules And Classes

### `GlassBox.numpandas`

`numpandas` is a minimal pandas-like tabular layer built on NumPy. It is designed for GlassBox ML workflows, not as a full pandas replacement.

Main classes and functions:

- `DataFrame`: 2D columnar table storing columns independently as arrays.
- `Series`: 1D labeled data container.
- `Index`: label container with label lookup through `get_loc`.
- `read_csv`, `read_json`, `read_excel`: file readers returning GlassBox `DataFrame` objects.

Important behavior:

- Copy-on-write style operations return new objects instead of mutating sources.
- Missing values are represented consistently with `np.nan`.
- No public `inplace` API is used.
- Supported operations include column selection, boolean masking, `loc`, `iloc`, `head`, `tail`, `sample`, `fillna`, `dropna`, `astype`, `apply`, `map`, `describe`, and NumPy conversion.
- Intentionally unsupported: `groupby`, joins/merges, pivot/melt/stack, MultiIndex, and datetime/time-series APIs.

### `GlassBox.eda`

The EDA package provides profiling, descriptive statistics, outlier detection, and plotting helpers for `numpandas` objects.

Main classes and objects:

- `DataProfiler`: builds data summaries and HTML diagnostic reports.
- `UnivariateStats`: computes univariate descriptive statistics.
- `IQR_OutlierDetector`: detects outliers using interquartile range logic.
- `PlotManager`: facade for common plots.
- `plot_manager`: shared `PlotManager` instance.
- Plotter classes: `HistPlotter`, `BoxPlotter`, `ScatterPlotter`, `MissingnessPlotter`, `CountPlotter`, `CorrelationMatrixPlotter`, and `PairPlotMatrixPlotter`.

### `GlassBox.preprocessing`

The preprocessing package follows scikit-learn-style `fit`, `transform`, and `fit_transform` conventions while working with GlassBox tabular structures.

Main classes and functions:

- `Transformer`: abstract base class for preprocessing transformers.
- `FunctionTransformer`: wraps custom transformation functions.
- `SimpleImputer`: imputes missing values with strategies such as mean, median, and most frequent.
- `StandardScaler`: standardizes features by removing the mean and scaling to unit variance.
- `MinMaxScaler`: scales values into a `[0, 1]` range.
- `RobustScaler`: scales values using IQR-based robust statistics.
- `OneHotEncoder`: one-hot encodes categorical columns.
- `OrdinalEncoder`: ordinal-encodes categorical columns.
- `LabelEncoder`: encodes target labels.
- `ColumnTransformer`: applies different transformers to selected columns.
- `make_column_transformer`: convenience factory for `ColumnTransformer`.
- `SMOTE`: synthetic minority oversampling for imbalanced datasets.
- Exceptions: `PreprocessingError`, `NotFittedError`, and `DimensionalityError`.

### `GlassBox.ml`

The ML package contains transparent implementations of traditional machine-learning algorithms and common metrics.

Main classes:

- `DecisionTree`: tree model for classification/regression-style tasks.
- `RandomForest`: ensemble tree model.
- `GaussianNaiveBayes`: probabilistic classifier.
- `KNNClassifier`: k-nearest-neighbors classifier.
- `KNNRegressor`: k-nearest-neighbors regressor.
- `LinearRegressionGD`: gradient-descent linear regression.
- `LogisticRegressionGD`: gradient-descent logistic regression.

Metrics and utilities:

- Classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`.
- Regression: `mean_absolute_error`, `mean_squared_error`, `r2_score`.
- Utility split helper: `train_test_split`.

### `GlassBox.split`

The split package provides standalone dataset splitting utilities that operate on GlassBox data structures.

Functions:

- `train_test_split`: returns train/test partitions.
- `train_validation_test_split`: returns train/validation/test partitions.

### `GlassBox.pipeline`

The pipeline package chains preprocessing steps and a final estimator into a single workflow.

Main class:

- `Pipeline`: sequentially runs named steps. Transformers are applied first, and the final estimator can be fitted, used for prediction, and scored.

### `GlassBox.optimization`

The optimization package provides cross-validation and hyperparameter search.

Main classes and functions:

- `KFold`: k-fold splitter.
- `cross_val_score`: evaluates an estimator across folds.
- `GridSearchCV`: exhaustive search over parameter grids.
- `RandomizedSearchCV`: sampled search over parameter spaces.

### `GlassBox.autofit`

AutoFit is the top-level AutoML orchestrator. It loads a CSV dataset, profiles it, detects the task type, preprocesses features, trains/tunes candidate models, evaluates them, builds a final pipeline, and optionally serializes the fitted model.

Main entry points:

- `autofit(csv_path, target_col, models=None, tuning=True, output_path="/results/best_model.pkl")`
- CLI: `python -m GlassBox.autofit.cli --data data.csv --target label --output /results/best_model.pkl`

Supporting modules:

- `core.py`: central `autofit()` workflow.
- `detect.py`: task detection for classification vs regression.
- `registry.py`: model registry and default search spaces.
- `report.py`: JSON-ready report construction.
- `cli.py`: command-line interface.

### `GlassBox.benchmark`

The benchmark package compares selected GlassBox operations against pandas and scikit-learn baselines.

Main pieces:

- `BenchmarkResult`: structured benchmark result record.
- `compare.py`: benchmark runner and report writer.
- CLI module: `python -m GlassBox.benchmark`.

The benchmark covers dataframe reductions, preprocessing, metrics, and selected model tasks.

## Installation

GlassBox targets Python 3.11+.

For local development from the repository root, install the packages in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt

python -m pip install -e packages/glassbox-numpandas
python -m pip install -e packages/glassbox-eda
python -m pip install -e packages/glassbox-preprocessing
python -m pip install -e packages/glassbox-ml
python -m pip install -e packages/glassbox-split
python -m pip install -e packages/glassbox-pipeline
python -m pip install -e packages/glassbox-optimization
python -m pip install -e packages/glassbox-benchmark
python -m pip install -e packages/glassbox-autofit
```

If the packages are already published or available in your package index, you can install the umbrella package:

```bash
python -m pip install glassbox-meta
```

## How To Use GlassBox

### 1. Create And Inspect Data

```python
import numpy as np

from GlassBox.numpandas import DataFrame, Series

df = DataFrame({
    "age": [20, 25, np.nan, 35],
    "income": [3000, 4200, 3900, 6100],
    "city": ["A", "B", "A", "C"],
})

target = Series([0, 1, 0, 1], name="target")

print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.describe())
print(df.head(2))
```

### 2. Select Rows And Columns

```python
ages = df["age"]
numeric = df[["age", "income"]]

first_age = df.iloc[0, 0]
city_for_second_row = df.loc[1, "city"]

mask = df["income"].to_numpy() > 3500
high_income = df[mask]
```

### 3. Handle Missing Values

```python
filled = df.fillna({"age": 0})
clean_rows = df.dropna(axis=0, how="any")
summary = df.describe()
```

### 4. Read And Write Files

```python
from GlassBox.numpandas import read_csv, read_json, read_excel

csv_df = read_csv("data.csv")
json_df = read_json("data.json")
excel_df = read_excel("data.xlsx")

csv_df.to_csv("out.csv", index=False)
csv_df.to_json("out.json")
csv_df.to_excel("out.xlsx")
```

### 5. Run EDA

```python
from GlassBox.eda import DataProfiler, plot_manager

profiler = DataProfiler(df)
profiler.generate_html_report("analysis_report.html")

plot_manager.histplot(df, column="age", bins=10)
plot_manager.boxplot(df, column="income")
plot_manager.missingness(df, title="Missing Values")
```

### 6. Preprocess Features

```python
from GlassBox.preprocessing import (
    ColumnTransformer,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
    make_column_transformer,
)

numeric_steps = make_column_transformer(
    (SimpleImputer(strategy="mean"), ["age", "income"]),
    (StandardScaler(), ["age", "income"]),
    remainder="drop",
)

X_numeric = numeric_steps.fit_transform(df)

categorical_steps = ColumnTransformer(
    transformers=[
        ("city_encoder", OneHotEncoder(), ["city"]),
    ],
    remainder="drop",
)

X_categorical = categorical_steps.fit_transform(df)
```

### 7. Train A Model

```python
from GlassBox.ml import RandomForest, accuracy_score
from GlassBox.numpandas import DataFrame, Series

X = DataFrame({
    "feature_1": [1.5, 2.3, 3.1, 4.0],
    "feature_2": [0.5, 1.2, 0.9, 2.1],
})
y = Series([0, 1, 0, 1])

model = RandomForest(n_estimators=25, max_depth=5)
model.fit(X, y)

predictions = model.predict(X)
print(accuracy_score(y, predictions))
```

### 8. Split A Dataset

```python
from GlassBox.split import train_test_split, train_validation_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

X_tr, X_val, X_te, y_tr, y_val, y_te = train_validation_test_split(
    X,
    y,
    test_size=0.15,
    val_size=0.15,
    random_state=42,
)
```

### 9. Build A Pipeline

```python
from GlassBox.pipeline import Pipeline
from GlassBox.ml import LogisticRegressionGD, accuracy_score
from GlassBox.preprocessing import SimpleImputer, StandardScaler

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegressionGD(learning_rate=0.1, n_iterations=2000)),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
score = accuracy_score(y_test, predictions)
```

### 10. Tune Hyperparameters

```python
from GlassBox.ml import RandomForest
from GlassBox.optimization import GridSearchCV, RandomizedSearchCV, cross_val_score

model = RandomForest()

param_grid = {
    "n_estimators": [10, 25, 50],
    "max_depth": [3, 5, None],
}

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)

scores = cross_val_score(search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy")
print(scores)
```

For larger parameter spaces, use randomized search:

```python
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=4,
    cv=5,
    scoring="accuracy",
    random_state=42,
)
random_search.fit(X_train, y_train)
```

### 11. Run AutoFit From Python

```python
from GlassBox.autofit import autofit

report, fitted_pipeline = autofit(
    "test_data/test_model.csv",
    target_col="target",
    models=None,
    tuning=True,
    output_path="results/best_model.pkl",
)

print(report["task"])
print(report["best_model"])
print(report["models"])
```

AutoFit returns:

- `report`: JSON-ready dictionary containing task type, EDA summary, preprocessing summary, model metrics, cross-validation scores, and the selected best model.
- `fitted_pipeline`: a fitted `GlassBox.pipeline.Pipeline` containing preprocessing steps and the best estimator.

### 12. Run AutoFit From The CLI

```bash
python -m GlassBox.autofit.cli \
  --data test_data/test_model.csv \
  --target target \
  --output results/best_model.pkl
```

The CLI prints the JSON report to stdout and writes a pickle artifact when `--output` is provided.

### 13. Run AutoFit In Docker

Build the image from the repository root:

```bash
docker build -t glassbox-env:latest .
```

Run AutoFit with local data and results mounted into the container:

```bash
docker run --rm \
  -v "${PWD}/test_data:/data" \
  -v "${PWD}/results:/results" \
  glassbox-env:latest \
  python -m GlassBox.autofit.cli --data /data/test_model.csv --target target --output /results/best_model.pkl
```

On PowerShell, use backticks for multi-line commands:

```powershell
docker run --rm `
  -v "${PWD}\test_data:/data" `
  -v "${PWD}\results:/results" `
  glassbox-env:latest `
  python -m GlassBox.autofit.cli --data /data/test_model.csv --target target --output /results/best_model.pkl
```

### 14. Run The Agent Integration

The `agent/` folder exposes GlassBox AutoFit as an agent tool.

```bash
cd agent
python run_agent.py
```

Example prompt:

```text
Analyze the 'test_model.csv' and predict the 'target' column.
```

The agent mounts `test_data` and `results`, runs the Dockerized AutoFit engine, and persists the fitted model to `results/best_model.pkl`.

## Testing

Run the quick end-to-end integrity test from the repository root:

```bash
python integrity_test.py
```

This script verifies the main training flow: local package imports, synthetic data creation, train/test splitting, preprocessing, pipeline execution, `GridSearchCV`, model fitting, and prediction.

Expected output is similar to:

```text
Testing pipeline end-to-end...
Best params: {'clf__learning_rate': 0.1}
Best score: 0.6125
Predictions len: 20
Test finished successfully!
```

The exact score can change if the generated data or tuning behavior changes, but the script should finish with `Test finished successfully!`.



## Benchmarking

GlassBox includes a benchmark package for comparing selected operations against pandas and scikit-learn. Install development dependencies first because the benchmark baselines require pandas and scikit-learn:

```bash
python -m pip install -r requirements-dev.txt
```

Run the benchmark from the repository root:

```bash
python -m GlassBox.benchmark --rows 5000 --cols 8 --repeats 5
```

You can also run the script directly:

```bash
python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py --rows 5000 --cols 8 --repeats 5
```

Benchmark outputs are written to:

- `packages/glassbox-benchmark/results/comparison_results.csv`
- `packages/glassbox-benchmark/docs/glassbox_comparison_report.md`

The checked-in report under `benchmark_outputs/bench_report.md` used 5000 rows, 8 numeric columns, and 5 repeats. Results from that local run:

| Group | Case | GlassBox seconds | pandas/sklearn seconds | Speedup | Winner |
|---|---:|---:|---:|---:|---|
| DataFrame | mean 5000x8 | 0.000241 | 0.000696 | 2.90x | GlassBox |
| DataFrame | sum 5000x8 | 0.000084 | 0.000434 | 5.14x | GlassBox |
| Metrics | accuracy_score n=5000 | 0.000026 | 0.000791 | 30.51x | GlassBox |
| Metrics | mean_squared_error n=5000 | 0.000030 | 0.000236 | 7.85x | GlassBox |
| Models | KNN predict 1200x8 | 0.053866 | 0.004159 | 0.08x | pandas/sklearn |
| Models | linear regression fit 5000x8 | 0.014015 | 0.008969 | 0.64x | pandas/sklearn |
| Models | logistic regression fit 5000x8 | 0.219417 | 0.061276 | 0.28x | pandas/sklearn |
| Preprocessing | MinMaxScaler 5000x8 | 0.002756 | 0.000830 | 0.30x | pandas/sklearn |

Correctness checks from the same run showed exact or near-exact agreement for deterministic operations:

- `DataFrame.mean`: max absolute difference `0`
- `DataFrame.sum`: max absolute difference `0`
- `MinMaxScaler`: max absolute difference `2.220446049e-16`
- `accuracy_score`: absolute difference `0`
- `mean_squared_error`: absolute difference `0`

Model comparisons should be interpreted as predictive-quality comparisons on the same generated datasets, not exact implementation equivalence. The benchmark report shows similar predictive quality for the compared model cases, while pandas/scikit-learn remains faster for the measured model training/prediction tasks. Runtime numbers are wall-clock measurements from a local machine, so rerun the benchmark before making final performance claims.

## Build Packages

Each package can be built independently:

```bash
python -m pip install --upgrade build
python -m build packages/glassbox-numpandas
python -m build packages/glassbox-eda
python -m build packages/glassbox-preprocessing
python -m build packages/glassbox-ml
python -m build packages/glassbox-split
python -m build packages/glassbox-pipeline
python -m build packages/glassbox-optimization
python -m build packages/glassbox-benchmark
python -m build packages/glassbox-autofit
python -m build packages/glassbox-meta
```

## Documentation Map

Additional package-specific documentation is available in:

- `docs/packaging-architecture.md`
- `packages/glassbox-numpandas/docs/manual.md`
- `packages/glassbox-eda/docs/README.md`
- `packages/glassbox-preprocessing/docs/README.md`
- `packages/glassbox-ml/docs/README.md`
- `packages/glassbox-split/docs/README.md`
- `packages/glassbox-pipeline/docs/README.md`
- `packages/glassbox-optimization/docs/README.md`
- `packages/glassbox-autofit/README.md`
- `dockerization_details.md`
- `autofit_nemoclaw.md`
- `packages/glassbox-benchmark/docs/glassbox_comparison_report.md`

