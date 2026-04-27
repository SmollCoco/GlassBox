# glassbox-eda

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![NumPy-only Core](https://img.shields.io/badge/core-NumPy--only-2f9e44) ![Part of GlassBox](https://img.shields.io/badge/ecosystem-GlassBox-0b7285)

`glassbox-eda` provides exploratory data analysis utilities for `GlassBox.numpandas` objects, including feature profiling, univariate statistics, IQR-based outlier detection, and a plotting facade for histograms, scatter plots, missingness maps, and correlation views.

## Installation

```bash
pip install glassbox-eda
```

## Import Example

```python
from GlassBox.eda import DataProfiler, plot_manager
```

## Minimal Usage

```python
import numpy as np
from GlassBox.numpandas import DataFrame
from GlassBox.eda import DataProfiler, IQR_OutlierDetector

df = DataFrame({
    "x1": np.array([1.0, 2.0, 2.5, np.nan, 100.0]),
    "x2": np.array([3.0, 3.2, 3.1, 3.4, 3.3]),
    "group": np.array(["A", "A", "B", "B", "B"], dtype=object),
})

profiler = DataProfiler(df)
profiler.compute_profile()
print(profiler.profile["x1"])

outliers = IQR_OutlierDetector(multiplier=1.5).fit(df).get_outlier_report(df)
print(outliers)
```

## API Inventory

| Class | Purpose |
|---|---|
| `DataProfiler` | Compute per-column profile summaries and export HTML reports. |
| `UnivariateStats` | Static/class methods for mean, median, mode, std, skewness, kurtosis. |
| `IQR_OutlierDetector` | Fit IQR bounds, report outliers, and cap outlier values. |
| `BasePlotter` | Abstract plotting base class. |
| `HistPlotter` | Histogram plotting implementation. |
| `BoxPlotter` | Boxplot plotting implementation. |
| `ScatterPlotter` | Two-feature scatter plotting implementation. |
| `MissingnessPlotter` | Missing-value heatmap plotting implementation. |
| `CountPlotter` | Categorical count plot implementation. |
| `CorrelationMatrixPlotter` | Correlation heatmap plotting implementation. |
| `PairPlotMatrixPlotter` | Pairwise plot matrix implementation. |
| `PlotManager` | Facade exposing standardized plotting methods. |
| `plot_manager` | Module-level shared `PlotManager` instance. |

## Repository

Main GlassBox GitHub repository: [https://github.com/SmollCoco/GlassBox](https://github.com/SmollCoco/GlassBox)
