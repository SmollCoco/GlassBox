import numpy as np
from typing import Any

from GlassBox.numpandas.core.series import Series
from GlassBox.numpandas.core.dataframe import DataFrame


def calc_mean(arr: np.ndarray) -> float:
    """Manually calculate the mean of an array."""
    valid_arr = arr[~np.isnan(arr)]
    if len(valid_arr) == 0:
        return np.nan
    return float(np.sum(valid_arr) / len(valid_arr))

def calc_median(arr: np.ndarray) -> float:
    """Manually calculate the median of an array."""
    valid_arr = np.sort(arr[~np.isnan(arr)])
    n = len(valid_arr)
    if n == 0:
        return np.nan
    mid = n // 2
    if n % 2 == 0:
        return float((valid_arr[mid - 1] + valid_arr[mid]) / 2.0)
    else:
        return float(valid_arr[mid])

def calc_mode(arr: np.ndarray) -> Any:
    """Manually calculate the mode of an array."""
    # Remove nans depending on data type
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr[~np.isnan(arr)]
    else:
        from GlassBox.numpandas.utils.dtypes import is_nan_value
        arr = np.array([x for x in arr if not is_nan_value(x)])
        
    if len(arr) == 0:
        return np.nan if np.issubdtype(arr.dtype, np.number) else ""
        
    unique_vals, counts = np.unique(arr, return_counts=True)
    max_count_idx = np.argmax(counts)
    return unique_vals[max_count_idx]

def calc_std(arr: np.ndarray) -> float:
    """Manually calculate the sample standard deviation."""
    valid_arr = arr[~np.isnan(arr)]
    n = len(valid_arr)
    if n <= 1:
        return 0.0
    mean_val = float(np.sum(valid_arr) / n)
    variance = np.sum((valid_arr - mean_val) ** 2) / (n - 1)
    return float(np.sqrt(variance))

def calc_skewness(arr: np.ndarray) -> float:
    """Manually calculate the skewness."""
    valid_arr = arr[~np.isnan(arr)]
    n = len(valid_arr)
    if n <= 2:
        return np.nan
        
    mean_val = calc_mean(valid_arr)
    std_val = calc_std(valid_arr)
    
    if std_val == 0:
        return 0.0
        
    m3 = np.sum((valid_arr - mean_val) ** 3) / n
    skew = m3 / (std_val ** 3)
    
    # Adjust for sample skewness
    skew = skew * np.sqrt(n * (n - 1)) / (n - 2)
    return float(skew)

def calc_kurtosis(arr: np.ndarray) -> float:
    """Manually calculate the sample kurtosis."""
    valid_arr = arr[~np.isnan(arr)]
    n = len(valid_arr)
    if n <= 3:
        return np.nan
        
    mean_val = calc_mean(valid_arr)
    std_val = calc_std(valid_arr)
    
    if std_val == 0:
        return 0.0
        
    m4 = np.sum((valid_arr - mean_val) ** 4) / n
    v = std_val ** 2
    
    # Excess kurtosis
    excess_kurt = (m4 / (v ** 2)) - 3.0
    # Adjust for sample kurtosis
    sample_kurt = (excess_kurt * (n - 1) * ((n + 1) / ((n - 2) * (n - 3)))) + (6 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))
    return float(sample_kurt)


class IQR_OutlierDetector:
    """Detect and cap outliers based on the Interquartile Range (IQR) method."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.lower_bounds_: dict[str, float] = {}
        self.upper_bounds_: dict[str, float] = {}

    def fit(self, X: DataFrame) -> "IQR_OutlierDetector":
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        
        for col in X.columns:
            if not np.issubdtype(X.dtypes[col], np.number):
                continue
                
            arr = X[col].to_numpy().astype(float, copy=False)
            valid_arr = arr[~np.isnan(arr)]
            if len(valid_arr) == 0:
                continue
                
            q1 = float(np.percentile(valid_arr, 25))
            q3 = float(np.percentile(valid_arr, 75))
            iqr = q3 - q1
            
            self.lower_bounds_[col] = q1 - (self.multiplier * iqr)
            self.upper_bounds_[col] = q3 + (self.multiplier * iqr)
            
        return self

    def get_outlier_report(self, X: DataFrame) -> dict[str, int]:
        """Return the number of outliers detected per column."""
        report = {}
        for col in X.columns:
            if col not in self.lower_bounds_:
                continue
            arr = X[col].to_numpy().astype(float, copy=False)
            valid_mask = ~np.isnan(arr)
            outliers = (arr[valid_mask] < self.lower_bounds_[col]) | (arr[valid_mask] > self.upper_bounds_[col])
            report[col] = int(np.sum(outliers))
        return report

    def cap_outliers(self, X: DataFrame) -> DataFrame:
        """Cap the detected outliers to the lower/upper bounds safely."""
        data = {}
        for col in X.columns:
            if col in self.lower_bounds_:
                arr = X[col].to_numpy().astype(float, copy=False).copy()
                valid_mask = ~np.isnan(arr)
                
                # Cap below lower bound
                lower = self.lower_bounds_[col]
                arr[valid_mask & (arr < lower)] = lower
                
                # Cap above upper bound
                upper = self.upper_bounds_[col]
                arr[valid_mask & (arr > upper)] = upper
                
                data[col] = arr
            else:
                data[col] = X[col].to_numpy()
                
        return DataFrame(data, columns=X.columns, index=X.index.to_list())
