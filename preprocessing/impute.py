import numpy as np

from numpandas.core.dataframe import DataFrame
from numpandas.core.series import Series
from numpandas.utils.dtypes import is_nan_value

from .base import Transformer
from .exceptions import NotFittedError


class SimpleImputer(Transformer):
    """Imputation transformer for completing missing values.
    
    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy.
        - If 'mean', then replace missing values using the mean along each column.
        - If 'median', then replace missing values using the median along each column.
        - If 'most_frequent', then replace missing using the most frequent value.
        - If 'constant', then replace missing values with fill_value.
    fill_value : any, optional
        When strategy == 'constant', fill_value is used to replace all missing values.
    imputation_indicator : bool, default=False
        If True, adds a binary column denoting if the value was originally missing.
        The name of the new column will be {column_name}_imputed.
    """

    def __init__(self, strategy: str = "mean", fill_value: any = None, imputation_indicator: bool = False):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputation_indicator = imputation_indicator
        
        self.statistics_: dict[str, any] = {}
        self.feature_names_in_: list[str] = []
        self._is_fitted = False

    def fit(self, X: DataFrame, y: Series | None = None) -> "SimpleImputer":
        self.statistics_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            arr = X[col].to_numpy()
            if self.strategy in ["mean", "median"]:
                if not np.issubdtype(X.dtypes[col], np.number):
                    raise TypeError(f"Cannot compute {self.strategy} on non-numeric column: {col}")
                numeric_arr = arr.astype(float, copy=False)
                if self.strategy == "mean":
                    self.statistics_[col] = float(np.nanmean(numeric_arr))
                else:
                    self.statistics_[col] = float(np.nanmedian(numeric_arr))
            
            elif self.strategy == "most_frequent":
                valid_values = [v for v in arr if not is_nan_value(v)]
                if not valid_values:
                    self.statistics_[col] = 0.0 if np.issubdtype(X.dtypes[col], np.number) else ""
                else:
                    values, counts = np.unique(valid_values, return_counts=True)
                    self.statistics_[col] = values[np.argmax(counts)]
                    
            elif self.strategy == "constant":
                self.statistics_[col] = self.fill_value
                
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
                
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        new_columns = []
        
        for col in X.columns:
            arr = X[col].to_numpy()
            
            # Identify missing values
            if np.issubdtype(arr.dtype, np.floating):
                mask = np.isnan(arr)
            else:
                mask = np.array([is_nan_value(v) for v in arr], dtype=bool)
                
            # Perform imputation
            filled_arr = arr.copy()
            if col in self.statistics_:
                filled_arr[mask] = self.statistics_[col]
            
            data[col] = filled_arr
            new_columns.append(col)
            
            # Add indicator if requested
            if self.imputation_indicator and col in self.feature_names_in_:
                indicator_col = f"{col}_imputed"
                data[indicator_col] = mask.astype(int)
                new_columns.append(indicator_col)
                
        return DataFrame(data, columns=new_columns, index=X.index.to_list())
