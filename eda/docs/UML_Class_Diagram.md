# EDA Package UML Class Diagram

```mermaid
classDiagram
    class BasePlotter {
        <<Abstract>>
        +plot(df: DataFrame, *args, **kwargs) None
        #_prepare_axes()
    }
    
    class HistPlotter {
        +plot(df, column, bins, title, ax)
    }
    class BoxPlotter {
        +plot(df, column, title, ax)
    }
    class ScatterPlotter {
        +plot(df, x, y, title, ax)
    }
    class MissingnessPlotter {
        +plot(df, title, ax)
    }
    
    class PlotManager {
        -_hist_plotter: HistPlotter
        -_box_plotter: BoxPlotter
        -_scatter_plotter: ScatterPlotter
        -_missingness_plotter: MissingnessPlotter
        +histplot(df, column, bins, title)
        +boxplot(df, column, title)
        +scatterplot(df, x, y, title)
        +missingness(df, title)
        +multiplot(df, columns, plot_type, cols)
    }
    
    class DataProfiler {
        +df: DataFrame
        +feature_types: dict
        +profile: dict
        +compute_profile()
        +generate_html_report(filepath)
    }
    
    class IQR_OutlierDetector {
        +multiplier: float
        +fit(X)
        +get_outlier_report(X)
        +cap_outliers(X)
    }
    
    BasePlotter <|-- HistPlotter
    BasePlotter <|-- BoxPlotter
    BasePlotter <|-- ScatterPlotter
    BasePlotter <|-- MissingnessPlotter
    PlotManager --> HistPlotter
    PlotManager --> BoxPlotter
    PlotManager --> ScatterPlotter
    PlotManager --> MissingnessPlotter
```

## Description
- **PlotManager**: A Facade that simplifies plotting operations for the user.
- **BasePlotter** and its implementations: They adhere to the Single Responsibility Principle, each handling exactly one type of plot.
- **DataProfiler**: Responsible for typing features and gathering pure text properties into an aesthetically styled HTML report.
- **IQR_OutlierDetector**: Calculates IQR thresholds and caps outliers.
