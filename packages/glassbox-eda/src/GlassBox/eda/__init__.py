"""Exploratory Data Analysis module for numpandas DataFrame.

This module provides tools for statistical profiling and data visualization.

Classes:
- DataProfiler: Generates statistical profiles and HTML reports.
- PlotManager: Facade for plotting distribution, relationships, and missingness.
- IQR_OutlierDetector: Detects outliers using Interquartile Range logic.

Functions/Instances:
- plot_manager: A global instance of PlotManager for convenient usage.
- calc_mean, calc_median, calc_mode, calc_std, calc_skewness, calc_kurtosis: Manual statistical calculations.
"""

from .plotter import (
    BasePlotter,
    HistPlotter,
    BoxPlotter,
    ScatterPlotter,
    MissingnessPlotter,
    PlotManager,
    plot_manager
)
from .profiler import DataProfiler
from .stats import (
    calc_mean,
    calc_median,
    calc_mode,
    calc_std,
    calc_skewness,
    calc_kurtosis,
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
    "IQR_OutlierDetector",
    "calc_mean",
    "calc_median",
    "calc_mode",
    "calc_std",
    "calc_skewness",
    "calc_kurtosis"
]
