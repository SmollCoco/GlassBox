import json
import numpy as np

from numpandas.core.dataframe import DataFrame
from numpandas.utils.dtypes import is_nan_value
from .stats import calc_mean, calc_median, calc_mode, calc_std, calc_skewness, calc_kurtosis, IQR_OutlierDetector


class DataProfiler:
    """Generates a statistical profile of a DataFrame and handles automatic typing."""
    
    def __init__(self, df: DataFrame):
        self.df = df
        self.feature_types: dict[str, str] = {}
        self.profile: dict[str, dict] = {}
        
    def _auto_type(self, col: str) -> str:
        """Distinguish between Numerical, Categorical, and Boolean data."""
        arr = self.df[col].to_numpy()
        dtype = self.df.dtypes[col]
        
        # Check explicit bool
        if dtype == bool:
            return "Boolean"
            
        # Extract valid values
        if np.issubdtype(dtype, np.floating):
            valid_arr = arr[~np.isnan(arr)]
        else:
            valid_arr = np.array([v for v in arr if not is_nan_value(v)])
            
        if len(valid_arr) == 0:
            return "Categorical"
            
        unique_vals = np.unique(valid_arr)
        
        # Check if implicitly boolean (e.g., 0/1 or True/False encoded as strings/ints)
        if len(unique_vals) <= 2:
            return "Boolean"
            
        if np.issubdtype(dtype, np.number):
            # We differentiate categorical encoded as integers
            if np.issubdtype(dtype, np.integer) and len(unique_vals) < 20:
                # Heuristic: less than 20 unique values might be categorical ordinal
                return "Categorical"
            return "Numerical"
            
        return "Categorical"

    def compute_profile(self) -> None:
        """Compute all statistical metrics for each feature."""
        self.feature_types = {}
        self.profile = {}
        
        # Calculate outliers
        outlier_detector = IQR_OutlierDetector().fit(self.df)
        outlier_counts = outlier_detector.get_outlier_report(self.df)
        
        for col in self.df.columns:
            arr = self.df[col].to_numpy()
            ftype = self._auto_type(col)
            self.feature_types[col] = ftype
            
            # Common stats
            if np.issubdtype(arr.dtype, np.floating):
                missing = int(np.sum(np.isnan(arr)))
            else:
                missing = int(np.sum([is_nan_value(x) for x in arr]))
                
            total = len(arr)
            missing_pct = (missing / total) * 100 if total > 0 else 0
            
            stats = {
                "type": ftype,
                "missing": missing,
                "missing_pct": round(missing_pct, 2),
            }
            
            if ftype == "Numerical":
                numeric_arr = arr.astype(float, copy=False)
                stats["mean"] = round(calc_mean(numeric_arr), 4)
                stats["median"] = round(calc_median(numeric_arr), 4)
                stats["std"] = round(calc_std(numeric_arr), 4)
                stats["min"] = round(float(np.nanmin(numeric_arr)) if missing < total else np.nan, 4)
                stats["max"] = round(float(np.nanmax(numeric_arr)) if missing < total else np.nan, 4)
                stats["skewness"] = round(calc_skewness(numeric_arr), 4)
                stats["kurtosis"] = round(calc_kurtosis(numeric_arr), 4)
                stats["outliers_iqr"] = outlier_counts.get(col, 0)
                
            elif ftype in ("Categorical", "Boolean"):
                stats["mode"] = str(calc_mode(arr))
                # Count distincts
                if np.issubdtype(arr.dtype, np.floating):
                    distinct = len(np.unique(arr[~np.isnan(arr)]))
                else:
                    distinct = len(np.unique([v for v in arr if not is_nan_value(v)]))
                stats["distinct_values"] = distinct
                
            self.profile[col] = stats

    def generate_html_report(self, filepath: str) -> None:
        """Export the computed profile to an aesthetically pleasing HTML file."""
        if not self.profile:
            self.compute_profile()
            
        css = """
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 10px; }
        h2 { color: #2980b9; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: 600; color: #34495e; }
        tr:hover { background-color: #f1f2f6; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
        .badge-numerical { background: #e1f5fe; color: #0277bd; }
        .badge-categorical { background: #f3e5f5; color: #7b1fa2; }
        .badge-boolean { background: #e8f5e9; color: #2e7d32; }
        """
        
        html = [
            "<!DOCTYPE html>",
            "<html>",
            f"<head><title>EDA Profile Report</title><style>{css}</style></head>",
            "<body>",
            "<div class='container'>",
            "<h1>Exploratory Data Analysis Report</h1>",
        ]
        
        for col, stats in self.profile.items():
            ftype = stats.get("type", "Unknown").lower()
            html.append(f"<h2>{col} <span class='badge badge-{ftype}'>{ftype.upper()}</span></h2>")
            html.append("<table><tbody>")
            
            for key, value in stats.items():
                if key == "type":
                    continue
                display_key = key.replace("_", " ").title()
                html.append(f"<tr><th>{display_key}</th><td>{value}</td></tr>")
                
            html.append("</tbody></table>")
            
        html.append("</div></body></html>")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
