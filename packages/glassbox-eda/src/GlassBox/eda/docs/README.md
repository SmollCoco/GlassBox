# EDA (Exploratory Data Analysis) Package

This package provides a comprehensive, highly modular suite to perform Exploratory Data Analysis directly on `numpandas` DataFrame structures using `matplotlib` and `seaborn`.

## Features
- **Plotting Facade (`PlotManager`)**: Use `plot_manager` to easily generate visuals (`.histplot()`, `.boxplot()`, `.scatterplot()`, `.missingness()`, `.multiplot()`) without manually orchestrating `matplotlib` subplots in user code. Under the hood, this enforces the Single Responsibility Principle.
- **Statistical Operations (`eda.stats`)**: Math routines manually implemented from scratch using only root `numpy` calculations (Mean, Median, Mode, Std, Skewness, Kurtosis). Contains an `IQR_OutlierDetector` that finds and caps anomalies.
- **DataProfiler**: Distinguishes feature families Automatically (Boolean vs Numerical vs Categorical), extracting statistics and outputting an aesthetically pleasing HTML standalone summary report.

## Usage

```python
from GlassBox.numpandas import DataFrame
from GlassBox.eda import plot_manager, DataProfiler

# Plotting distribution
plot_manager.histplot(df, 'age')

# Analyzing Missingness across the whole dataset
plot_manager.missingness(df, title="Missing nulls map")

# Generate HTML statistical summary
profiler = DataProfiler(df)
profiler.generate_html_report('analysis_report.html')
```

## Structure
- `eda/plotter.py`: Implementations of visualizations.
- `eda/stats.py`: Statistics and Mathematics.
- `eda/profiler.py`: HTML report generation and typing.
- `eda/UML_Class_Diagram.md`: Class architecture.
