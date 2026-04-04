import numpy as np

from numpandas.core.dataframe import DataFrame
from numpandas.core.series import Series

from .base import Transformer
from .exceptions import NotFittedError


class StandardScaler(Transformer):
    """Standardize features by removing the mean and scaling to unit variance."""
    
    def __init__(self):
        self.mean_: dict[str, float] = {}
        self.var_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self._is_fitted = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: DataFrame, y: Series | None = None) -> "StandardScaler":
        self.mean_ = {}
        self.var_ = {}
        self.scale_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            if not np.issubdtype(X.dtypes[col], np.number):
                raise TypeError(f"Cannot scale non-numeric column: {col}")
            arr = X[col].to_numpy().astype(float, copy=False)
            col_mean = float(np.nanmean(arr))
            col_var = float(np.nanvar(arr, ddof=1)) if len(arr) > 1 else 0.0
            col_std = np.sqrt(col_var)
            
            self.mean_[col] = col_mean
            self.var_[col] = col_var
            self.scale_[col] = col_std if col_std != 0.0 else 1.0
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        for col in X.columns:
            if col in self.feature_names_in_:
                arr = X[col].to_numpy().astype(float, copy=False)
                scaled_arr = (arr - self.mean_[col]) / self.scale_[col]
                data[col] = scaled_arr
            else:
                data[col] = X[col].to_numpy()
                
        return DataFrame(data, columns=X.columns, index=X.index.to_list())


class MinMaxScaler(Transformer):
    """Transform features by scaling each feature to a given range (default 0, 1)."""
    
    def __init__(self):
        self.data_min_: dict[str, float] = {}
        self.data_max_: dict[str, float] = {}
        self.data_range_: dict[str, float] = {}
        self._is_fitted = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: DataFrame, y: Series | None = None) -> "MinMaxScaler":
        self.data_min_ = {}
        self.data_max_ = {}
        self.data_range_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            if not np.issubdtype(X.dtypes[col], np.number):
                raise TypeError(f"Cannot scale non-numeric column: {col}")
            arr = X[col].to_numpy().astype(float, copy=False)
            d_min = float(np.nanmin(arr))
            d_max = float(np.nanmax(arr))
            d_range = d_max - d_min
            
            self.data_min_[col] = d_min
            self.data_max_[col] = d_max
            self.data_range_[col] = d_range if d_range != 0.0 else 1.0
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        for col in X.columns:
            if col in self.feature_names_in_:
                arr = X[col].to_numpy().astype(float, copy=False)
                scaled_arr = (arr - self.data_min_[col]) / self.data_range_[col]
                data[col] = scaled_arr
            else:
                data[col] = X[col].to_numpy()
                
        return DataFrame(data, columns=X.columns, index=X.index.to_list())


class RobustScaler(Transformer):
    """Scale features using statistics that are robust to outliers (IQR)."""
    
    def __init__(self):
        self.center_: dict[str, float] = {}
        self.scale_: dict[str, float] = {}
        self._is_fitted = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: DataFrame, y: Series | None = None) -> "RobustScaler":
        self.center_ = {}
        self.scale_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            if not np.issubdtype(X.dtypes[col], np.number):
                raise TypeError(f"Cannot scale non-numeric column: {col}")
            arr = X[col].to_numpy().astype(float, copy=False)
            valid_arr = arr[~np.isnan(arr)]
            if len(valid_arr) == 0:
                self.center_[col] = 0.0
                self.scale_[col] = 1.0
                continue
                
            q1 = float(np.percentile(valid_arr, 25))
            median = float(np.percentile(valid_arr, 50))
            q3 = float(np.percentile(valid_arr, 75))
            iqr = q3 - q1
            
            self.center_[col] = median
            self.scale_[col] = iqr if iqr != 0.0 else 1.0
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        for col in X.columns:
            if col in self.feature_names_in_:
                arr = X[col].to_numpy().astype(float, copy=False)
                scaled_arr = (arr - self.center_[col]) / self.scale_[col]
                data[col] = scaled_arr
            else:
                data[col] = X[col].to_numpy()
                
        return DataFrame(data, columns=X.columns, index=X.index.to_list())
