import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABC, abstractmethod

from numpandas.core.dataframe import DataFrame


class BasePlotter(ABC):
    """Abstract base class for all plotters following Single Responsibility Principle."""
    
    @abstractmethod
    def plot(self, df: DataFrame, *args, **kwargs) -> None:
        """Execute the plot operation."""
        pass

    def _prepare_axes(self, ax=None, figsize=(10, 6)):
        """Helper to prepare matplotlib axes if none provided."""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        return fig, ax


class HistPlotter(BasePlotter):
    """Class responsible for plotting histograms."""
    
    def plot(self, df: DataFrame, column: str, bins: int = 30, title: str = None, ax=None) -> None:
        fig, ax = self._prepare_axes(ax)
        arr = df[column].to_numpy()
        
        if np.issubdtype(df.dtypes[column], np.floating):
            valid_data = arr[~np.isnan(arr)]
        else:
            # If not float, let seaborn handle it or convert
            valid_data = arr
            
        sns.histplot(valid_data, bins=bins, kde=True, ax=ax, color='skyblue')
        
        ax.set_title(title or f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        if ax.figure is fig: # meaning we created it
            plt.tight_layout()
            plt.show()


class BoxPlotter(BasePlotter):
    """Class responsible for plotting boxplots to identify outliers."""
    
    def plot(self, df: DataFrame, column: str, title: str = None, ax=None) -> None:
        fig, ax = self._prepare_axes(ax)
        arr = df[column].to_numpy()
        
        if np.issubdtype(df.dtypes[column], np.floating):
            valid_data = arr[~np.isnan(arr)]
        else:
            valid_data = arr
            
        sns.boxplot(x=valid_data, ax=ax, color='lightgreen')
        
        ax.set_title(title or f"Boxplot of {column}")
        ax.set_xlabel(column)
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class ScatterPlotter(BasePlotter):
    """Class responsible for plotting scatter plots between two specific columns."""
    
    def plot(self, df: DataFrame, x: str, y: str, title: str = None, ax=None) -> None:
        fig, ax = self._prepare_axes(ax)
        x_arr = df[x].to_numpy()
        y_arr = df[y].to_numpy()
        
        sns.scatterplot(x=x_arr, y=y_arr, ax=ax, alpha=0.7)
        
        ax.set_title(title or f"Scatter plot: {x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class MissingnessPlotter(BasePlotter):
    """Class responsible for plotting a missingness map (heatmap of nulls)."""
    
    def plot(self, df: DataFrame, title: str = None, ax=None) -> None:
        fig, ax = self._prepare_axes(ax, figsize=(12, 8))
        
        # Get missing boolean dataframe
        isna_df = df.isna()
        
        # Convert to boolean 2D numpy array for seaborn
        mask = isna_df.to_numpy()
        
        sns.heatmap(mask, yticklabels=False, cbar=False, cmap='viridis', ax=ax)
        
        ax.set_xticks(np.arange(len(df.columns)) + 0.5)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        
        ax.set_title(title or "Missingness Map (Yellow = Missing)")
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class PlotManager:
    """Facade for managing and standardizing EDA plots."""
    
    def __init__(self):
        self._hist_plotter = HistPlotter()
        self._box_plotter = BoxPlotter()
        self._scatter_plotter = ScatterPlotter()
        self._missingness_plotter = MissingnessPlotter()
        
    def histplot(self, df: DataFrame, column: str, bins: int = 30, title: str = None) -> None:
        """Plot a histogram for a column."""
        self._hist_plotter.plot(df, column=column, bins=bins, title=title)
        
    def boxplot(self, df: DataFrame, column: str, title: str = None) -> None:
        """Plot a boxplot for a column."""
        self._box_plotter.plot(df, column=column, title=title)
        
    def scatterplot(self, df: DataFrame, x: str, y: str, title: str = None) -> None:
        """Plot a scatterplot between two columns."""
        self._scatter_plotter.plot(df, x=x, y=y, title=title)
        
    def missingness(self, df: DataFrame, title: str = None) -> None:
        """Plot a missingness map for the entire DataFrame."""
        self._missingness_plotter.plot(df, title=title)
        
    def multiplot(self, df: DataFrame, columns: list[str], plot_type: str = "hist", cols: int = 2) -> None:
        """Plot multiple plots in one panel."""
        n = len(columns)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        
        axes_flat = np.array(axes).flatten()
        
        for i, col in enumerate(columns):
            ax = axes_flat[i]
            if plot_type == "hist":
                self._hist_plotter.plot(df, col, ax=ax)
            elif plot_type == "box":
                self._box_plotter.plot(df, col, ax=ax)
            else:
                ax.set_title(f"Unsupported plot type: {plot_type}")
                
        # Hide empty subplots
        for j in range(len(columns), len(axes_flat)):
            axes_flat[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()

# Global instance for easy access
plot_manager = PlotManager()
