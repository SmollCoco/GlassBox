import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series
from GlassBox.numpandas.utils.dtypes import is_nan_value

from .base import Transformer
from .exceptions import NotFittedError


class OneHotEncoder(Transformer):
    """Encode categorical features as a one-hot numeric array.
    
    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default='ignore'
        Whether to raise an error or ignore if an unknown categorical feature is 
        present during transform (default is to ignore and encode as all zeros).
    """

    def __init__(self, handle_unknown: str = "ignore"):
        self.handle_unknown = handle_unknown
        self.categories_: dict[str, list[any]] = {}
        self.feature_names_in_: list[str] = []
        self._is_fitted = False

    def fit(self, X: DataFrame, y: Series | None = None) -> "OneHotEncoder":
        self.categories_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            arr = X[col].to_numpy()
            valid_values = [v for v in arr if not is_nan_value(v)]
            unique_vals = list(np.unique(valid_values))
            
            # Convert values to strings for consistent identification, except for numbers if preferred,
            # but string representation is safer for categorical features
            unique_vals.sort(key=lambda x: str(x))
            self.categories_[col] = unique_vals
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        new_columns = []
        
        for col in X.columns:
            if col not in self.feature_names_in_:
                data[col] = X[col].to_numpy()
                new_columns.append(col)
                continue
                
            arr = X[col].to_numpy()
            cats = self.categories_[col]
            
            # Create a zero-initialized array for each category
            n_samples = len(arr)
            for cat in cats:
                cat_col_name = f"{col}_{cat}"
                data[cat_col_name] = np.zeros(n_samples, dtype=int)
                new_columns.append(cat_col_name)
                
            # Populate 1s where matched
            for i, val in enumerate(arr):
                if is_nan_value(val):
                    continue
                if val in cats:
                    cat_col_name = f"{col}_{val}"
                    data[cat_col_name][i] = 1
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Found unknown category {val} in column {col} during transform.")
                    # If 'ignore', it remains 0 across all dummy columns for this feature
                    
        return DataFrame(data, columns=new_columns, index=X.index.to_list())


class OrdinalEncoder(Transformer):
    """Encode categorical features as an integer array.
    
    This estimator transforms 2D datasets (like DataFrame) where each column 
    is a categorical feature.
    """

    def __init__(self, handle_unknown: str = "error", unknown_value: int = -1):
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.categories_: dict[str, list[any]] = {}
        self.feature_names_in_: list[str] = []
        self._is_fitted = False

    def fit(self, X: DataFrame, y: Series | None = None) -> "OrdinalEncoder":
        self.categories_ = {}
        self.feature_names_in_ = X.columns
        
        for col in self.feature_names_in_:
            arr = X[col].to_numpy()
            valid_values = [v for v in arr if not is_nan_value(v)]
            unique_vals = list(np.unique(valid_values))
            unique_vals.sort(key=lambda x: str(x))
            self.categories_[col] = unique_vals
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        data = {}
        for col in X.columns:
            if col not in self.feature_names_in_:
                data[col] = X[col].to_numpy()
                continue
                
            arr = X[col].to_numpy()
            cats = self.categories_[col]
            cat_to_int = {cat: i for i, cat in enumerate(cats)}
            
            encoded_arr = np.zeros(len(arr), dtype=float)
            for i, val in enumerate(arr):
                if is_nan_value(val):
                    encoded_arr[i] = np.nan
                elif val in cat_to_int:
                    encoded_arr[i] = cat_to_int[val]
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Found unknown category {val} in column {col} during transform.")
                    encoded_arr[i] = self.unknown_value
                    
            data[col] = encoded_arr
            
        return DataFrame(data, columns=X.columns, index=X.index.to_list())


class LabelEncoder:
    """Encode target labels with value between 0 and n_classes-1.
    
    Note: LabelEncoder should be used to encode target values (i.e. y), 
    and not the input X. For X features, use OrdinalEncoder.
    """

    def __init__(self):
        self.classes_: np.ndarray | None = None

    def fit(self, y: Series | np.ndarray) -> "LabelEncoder":
        arr = y.to_numpy() if isinstance(y, Series) else np.asarray(y)
        arr = arr.reshape(-1)
        valid_values = [v for v in arr if not is_nan_value(v)]
        self.classes_ = np.unique(valid_values)
        self.classes_.sort()
        return self

    def transform(self, y: Series | np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise NotFittedError()
            
        arr = y.to_numpy() if isinstance(y, Series) else np.asarray(y)
        arr = arr.reshape(-1)
        
        class_to_int = {cat: i for i, cat in enumerate(self.classes_)}
        
        encoded_arr = np.zeros(len(arr), dtype=float)
        for i, val in enumerate(arr):
            if is_nan_value(val):
                encoded_arr[i] = np.nan
            elif val in class_to_int:
                encoded_arr[i] = class_to_int[val]
            else:
                raise ValueError(f"Found unknown label {val} during transform.")
                
        return encoded_arr

    def fit_transform(self, y: Series | np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)
