"""Machine learning preprocessing module for numpandas DataFrame.

This module provides transformers, scalers, and encoders to preprocess data.

Classes:
- Transformer: Base class for all transformers.
- FunctionTransformer: Transforms data using a custom function.
- SimpleImputer: Imputes missing values.
- StandardScaler: Standardizes features by removing mean and scaling to unit variance.
- MinMaxScaler: Scales features to a [0, 1] range.
- RobustScaler: Scales features using robust statistics (IQR).
- OneHotEncoder: Encodes categorical features as a one-hot numeric array.
- OrdinalEncoder: Encodes categorical features as ordinal integers.
- LabelEncoder: Encodes target labels with values between 0 and n_classes-1.
- ColumnTransformer: Applies transformers to specific columns in a dataset.
- SMOTE: Synthetic Minority Over-sampling Technique for imbalanced datasets.

Functions:
- make_column_transformer: Factory function for creating ColumnTransformers.
"""

from .base import Transformer
from .compose import ColumnTransformer, make_column_transformer
from .encode import LabelEncoder, OneHotEncoder, OrdinalEncoder
from .exceptions import DimensionalityError, NotFittedError, PreprocessingError
from .impute import SimpleImputer
from .scale import MinMaxScaler, RobustScaler, StandardScaler
from .transform import FunctionTransformer
from .smote import SMOTE

__all__ = [
    "Transformer",
    "FunctionTransformer",
    "SimpleImputer",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "ColumnTransformer",
    "make_column_transformer",
    "PreprocessingError",
    "NotFittedError",
    "DimensionalityError",
    "SMOTE",
]
