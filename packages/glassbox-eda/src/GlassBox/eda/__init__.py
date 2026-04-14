"""Exploratory Data Analysis module for numpandas DataFrame.

This module provides tools for statistical profiling and data visualization.

Classes:
- DataProfiler: Generates statistical profiles and HTML reports.
- PlotManager: Facade for plotting distribution, relationships, and missingness.
- IQR_OutlierDetector: Detects outliers using Interquartile Range logic.
- UnivariateStats: Univariate statistical calculations.

Functions/Instances:
- plot_manager: A global instance of PlotManager for convenient usage.
"""

from .plotter import (
    BasePlotter,
    HistPlotter,
    BoxPlotter,
    ScatterPlotter,
    MissingnessPlotter,
    CountPlotter,
    CorrelationMatrixPlotter,
    PairPlotMatrixPlotter,
    PlotManager,
    plot_manager
)
from .profiler import DataProfiler
from .stats import (
    UnivariateStats,
    IQR_OutlierDetector
)

__all__ = [
    "DataProfiler",
    "PlotManager",
    "plot_manager",
    "BasePlotter",
    "HistPlotter",
    "BoxPlotter",
    "ScatterPlotter",
    "MissingnessPlotter",
    "CountPlotter",
    "CorrelationMatrixPlotter",
    "PairPlotMatrixPlotter",
    "IQR_OutlierDetector",
    "UnivariateStats"
]
