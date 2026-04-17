# glassbox-autofit

`glassbox-autofit` is the top-level AutoML orchestrator for the GlassBox monorepo.
It runs an end-to-end workflow:

1. load CSV data
2. profile data (EDA)
3. detect task type (classification/regression)
4. preprocess features
5. train and optionally tune models
6. evaluate models
7. return a JSON-ready report dictionary

## Installation

```bash
pip install glassbox-autofit
```

## Minimal Usage

```python
from GlassBox.autofit import autofit

report = autofit("data.csv", target_col="label")
print(report["best_model"])
```

## Parameters

- `csv_path: str` - Path to input CSV file.
- `target_col: str` - Name of the target column.
- `models: list[str] | None = None` - Optional list of model names to train.
  - If `None`, all models for the detected task are used.
- `tuning: bool = True` - If `True`, runs `RandomizedSearchCV` for each model.

## JSON Output Structure

The returned dictionary has this shape:

```json
{
  "task": "classification",
  "eda": {
    "shape": [100, 5],
    "dtypes": {"feature1": "float64"},
    "missing_per_column": {"feature1": 0},
    "outliers_detected": {"feature1": 2},
    "correlations": {"feature1_feature2": 0.42}
  },
  "preprocessing": {
    "imputer": "mean+most_frequent",
    "scaler": "StandardScaler",
    "encoders_applied": ["OneHotEncoder"]
  },
  "models": [
    {
      "name": "RandomForestClassifier",
      "best_params": {},
      "metrics": {
        "accuracy": 0.9,
        "precision": 0.88,
        "recall": 0.91,
        "f1": 0.89
      },
      "cv_score": 0.87
    }
  ],
  "best_model": "RandomForestClassifier"
}
```
