# GlassBox EDA

The EDA package powers Exploratory Data Analysis by integrating Automated Profiling alongside elegant Facade-based plotting mechanisms extending generic `seaborn` and `matplotlib` functionalities gracefully scaled onto `numpandas` objects.

## Key Features
- **DataProfiler**: Distinguishes types internally (Booleans vs Categorical vs Numerical), compiling summary modes, skewness factors, outlies boundaries, and deploying clean HTML diagnostic profiles.
- **PlotManager**: Aggregates plot instantiations using Single-Responsibility plotter subcomponents correctly mapped under the hood.

## API Usage

```python
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.eda.plotter import plot_manager
from GlassBox.eda.profiler import DataProfiler

df = DataFrame({"age": [20, 25, 30], "height": [170, 180, 175]})

# Generate standalone visual
plot_manager.histplot(df, column="age", bins=10)
plot_manager.missingness(df, title="Project Integrity Map")

# Generate standalone HTML file with aggregated summary statistics
profiler = DataProfiler(df)
profiler.generate_html_report('analysis_report.html')
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.