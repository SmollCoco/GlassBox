# glassbox-preprocessing

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![NumPy-only Core](https://img.shields.io/badge/core-NumPy--only-2f9e44) ![Part of GlassBox](https://img.shields.io/badge/ecosystem-GlassBox-0b7285)

`glassbox-preprocessing` implements scikit-learn-style preprocessing for GlassBox data structures, covering imputation, scaling, categorical encoding, column-wise composition, custom function transforms, and class balancing with SMOTE.

## Installation

```bash
pip install glassbox-preprocessing
```

## Import Example

```python
from GlassBox.preprocessing import SimpleImputer, StandardScaler
```

## Minimal Usage

```python
import numpy as np
from GlassBox.numpandas import DataFrame
from GlassBox.preprocessing import SimpleImputer, StandardScaler

X = DataFrame(
    {
        "age": np.array([20.0, np.nan, 40.0]),
        "income": np.array([2500.0, 1800.0, 3200.0]),
    }
)

X_filled = SimpleImputer(strategy="mean").fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_filled)
print(X_scaled.to_numpy())
```

## API Inventory

| Class | Purpose |
|---|---|
| `Transformer` | Abstract base class for preprocessing transformers. |
| `FunctionTransformer` | Wrap arbitrary callable transformations. |
| `SimpleImputer` | Fill missing values via mean/median/most_frequent/constant strategies. |
| `StandardScaler` | Standardize numeric features using mean and variance. |
| `MinMaxScaler` | Scale numeric features to a fixed range (default $[0,1]$). |
| `RobustScaler` | Scale numeric features with robust (IQR-based) statistics. |
| `OneHotEncoder` | Expand categorical columns into one-hot indicator columns. |
| `OrdinalEncoder` | Encode categorical columns as ordinal integer values. |
| `LabelEncoder` | Encode target labels to integer class IDs. |
| `ColumnTransformer` | Apply different transformers to different column subsets. |
| `make_column_transformer(*transformers: tuple[Transformer, list[str]], remainder: str = "drop") -> ColumnTransformer` | Factory helper to build a `ColumnTransformer`. |
| `SMOTE` | Perform synthetic minority over-sampling for imbalanced targets. |
| `PreprocessingError` | Base exception for preprocessing failures. |
| `NotFittedError` | Raised when transform/predict-like operations are called before fit. |
| `DimensionalityError` | Raised for unexpected shape/dimensionality conditions. |

## Repository

Main GlassBox GitHub repository: [https://github.com/SmollCoco/GlassBox](https://github.com/SmollCoco/GlassBox)
