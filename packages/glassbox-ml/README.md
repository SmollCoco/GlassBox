# glassbox-ml

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![NumPy-only Core](https://img.shields.io/badge/core-NumPy--only-2f9e44) ![Part of GlassBox](https://img.shields.io/badge/ecosystem-GlassBox-0b7285)

`glassbox-ml` contains transparent, NumPy-first machine learning building blocks for classification and regression, including tree models, random forests, KNN, linear/logistic gradient-descent models, metrics, and utility helpers such as dataset splitting.

## Installation

```bash
pip install glassbox-ml
```

## Import Example

```python
from GlassBox.ml import GaussianNaiveBayes, train_test_split, accuracy_score
```

## Minimal Usage

```python
import numpy as np
from GlassBox.ml import GaussianNaiveBayes, train_test_split, accuracy_score

X = np.array([
    [1.0, 2.0],
    [1.2, 1.8],
    [3.5, 4.0],
    [3.8, 3.6],
    [0.8, 2.2],
    [3.2, 3.9],
])
y = np.array([0, 0, 1, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = GaussianNaiveBayes(var_smoothing=1e-9).fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

## API Inventory

| Class | Purpose |
|---|---|
| `DecisionTree` | Tree learner for classification or regression. |
| `_Node` | Internal tree node representation used by `DecisionTree`. |
| `RandomForest` | Bagging + feature-subspace ensemble of decision trees. |
| `GaussianNaiveBayes` | Gaussian Naive Bayes classifier. |
| `_BaseKNN` | Internal KNN shared implementation. |
| `KNNClassifier` | K-nearest neighbors classifier. |
| `KNNRegressor` | K-nearest neighbors regressor. |
| `LinearRegressionGD` | Batch gradient-descent linear regression. |
| `LogisticRegressionGD` | Batch gradient-descent binary logistic regression. |
| `_binary_targets(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]` | Internal binary-target validation helper for metrics. |
| `accuracy_score(y_true: Any, y_pred: Any) -> float` | Classification accuracy metric. |
| `precision_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float` | Binary precision metric. |
| `recall_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float` | Binary recall metric. |
| `f1_score(y_true: Any, y_pred: Any, positive_label: Any = 1) -> float` | Binary F1 metric. |
| `mean_absolute_error(y_true: Any, y_pred: Any) -> float` | Regression MAE metric. |
| `mean_squared_error(y_true: Any, y_pred: Any) -> float` | Regression MSE metric. |
| `r2_score(y_true: Any, y_pred: Any) -> float` | Regression $R^2$ metric. |
| `to_numpy_2d(data: Any, *, dtype: Any = float, name: str = "X") -> np.ndarray` | Convert supported inputs to validated 2D NumPy arrays. |
| `to_numpy_1d(data: Any, *, dtype: Any | None = float, name: str = "y") -> np.ndarray` | Convert supported inputs to validated 1D NumPy arrays. |
| `validate_same_length(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]` | Ensure feature/target row counts match. |
| `add_intercept(X: np.ndarray) -> np.ndarray` | Append bias/intercept column. |
| `train_test_split(X: Any, y: Any, test_size: float = 0.2, random_state: int | None = None, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` | Split arrays into training and test partitions. |

## Repository

Main GlassBox GitHub repository: [https://github.com/SmollCoco/GlassBox](https://github.com/SmollCoco/GlassBox)
