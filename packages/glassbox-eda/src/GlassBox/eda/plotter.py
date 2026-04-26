import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from abc import ABC, abstractmethod

from GlassBox.numpandas.core.dataframe import DataFrame


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
    
    def plot(self, df: DataFrame, column: str, bins: int = 30, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax)
        arr = df[column].to_numpy()
        
        if np.issubdtype(df.dtypes[column], np.floating):
            valid_data = arr[~np.isnan(arr)]
        else:
            valid_data = arr
            
        # Extract kde from kwargs with default False instead of always True (user requested control)
        kde = kwargs.pop('kde', False)
        
        sns.histplot(valid_data, bins=bins, kde=kde, ax=ax, color=kwargs.pop('color', 'skyblue'), **kwargs)
        
        ax.set_title(title or f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        if ax.figure is fig: # meaning we created it
            plt.tight_layout()
            plt.show()


class BoxPlotter(BasePlotter):
    """Class responsible for plotting boxplots to identify outliers."""
    
    def plot(self, df: DataFrame, column: str, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax)
        arr = df[column].to_numpy()
        
        if np.issubdtype(df.dtypes[column], np.floating):
            valid_data = arr[~np.isnan(arr)]
        else:
            valid_data = arr
            
        sns.boxplot(x=valid_data, ax=ax, color=kwargs.pop('color', 'lightgreen'), **kwargs)
        
        ax.set_title(title or f"Boxplot of {column}")
        ax.set_xlabel(column)
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class ScatterPlotter(BasePlotter):
    """Class responsible for plotting scatter plots between two specific columns."""
    
    def plot(self, df: DataFrame, x: str, y: str, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax)
        x_arr = df[x].to_numpy()
        y_arr = df[y].to_numpy()
        
        alpha = kwargs.pop('alpha', 0.7)
        sns.scatterplot(x=x_arr, y=y_arr, ax=ax, alpha=alpha, **kwargs)
        
        ax.set_title(title or f"Scatter plot: {x} vs {y}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class MissingnessPlotter(BasePlotter):
    """Class responsible for plotting a missingness map (heatmap of nulls)."""
    
    def plot(self, df: DataFrame, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax, figsize=(12, 8))
        
        isna_df = df.isna()
        mask = isna_df.to_numpy()
        
        cmap = kwargs.pop('cmap', 'viridis')
        sns.heatmap(mask, yticklabels=False, cbar=False, cmap=cmap, ax=ax, **kwargs)
        
        ax.set_xticks(np.arange(len(df.columns)) + 0.5)
        ax.set_xticklabels(df.columns, rotation=45, ha='right')
        
        ax.set_title(title or "Missingness Map (Yellow = Missing)")
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class CountPlotter(BasePlotter):
    """Class responsible for plotting count plots for categorical data."""
    
    def plot(self, df: DataFrame, column: str, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax)
        arr = df[column].to_numpy()
        
        sns.countplot(x=arr, ax=ax, palette=kwargs.pop('palette', 'mako'), **kwargs)
        
        ax.set_title(title or f"Count plot of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        
        # Rotate x labels if there are many categories
        if hasattr(ax, 'get_xticklabels'):
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class CorrelationMatrixPlotter(BasePlotter):
    """Class responsible for plotting the correlation matrix heatmap."""
    
    def plot(self, df: DataFrame, title: str = None, ax=None, **kwargs) -> None:
        fig, ax = self._prepare_axes(ax, figsize=(10, 8))
        
        # Only select numerical columns
        numerical_cols = [c for c in df.columns if np.issubdtype(df.dtypes[c], np.number)]
        if not numerical_cols:
            print("No numerical columns available for correlation matrix.")
            return
            
        corr_data = {}
        for col in numerical_cols:
            corr_data[col] = df[col].to_numpy().astype(float)
            
        # Calculate pearson correlation matrix manually (handling nan)
        n_cols = len(numerical_cols)
        corr_matrix = np.eye(n_cols)
        
        for i in range(n_cols):
            for j in range(i+1, n_cols):
                arri = corr_data[numerical_cols[i]]
                arrj = corr_data[numerical_cols[j]]
                
                # Use only rows where both are not nan
                valid = ~(np.isnan(arri) | np.isnan(arrj))
                if np.sum(valid) > 1:
                    vi = arri[valid]
                    vj = arrj[valid]
                    std_i = np.std(vi)
                    std_j = np.std(vj)
                    if std_i > 0 and std_j > 0:
                        cov = np.mean((vi - np.mean(vi)) * (vj - np.mean(vj)))
                        corr = cov / (std_i * std_j)
                    else:
                        corr = 0.0
                else:
                    corr = 0.0
                    
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        cmap = kwargs.pop('cmap', 'coolwarm')
        annot = kwargs.pop('annot', True)
        sns.heatmap(corr_matrix, xticklabels=numerical_cols, yticklabels=numerical_cols, 
                    cmap=cmap, annot=annot, ax=ax, vmin=-1, vmax=1, **kwargs)
        
        ax.set_title(title or "Correlation Matrix")
        if ax.figure is fig:
            plt.tight_layout()
            plt.show()


class PairPlotMatrixPlotter(BasePlotter):
    """Class responsible for plotting pair plots."""
    
    def plot(self, df: DataFrame, title: str = None, **kwargs) -> None:
        numerical_cols = [c for c in df.columns if np.issubdtype(df.dtypes[c], np.number)]
        if not numerical_cols:
            print("No numerical columns to pairplot.")
            return

        # Extract pairplot-like options while keeping method signature unchanged.
        hue = kwargs.pop('hue', None)
        palette = kwargs.pop('palette', 'deep')
        diag_kind = kwargs.pop('diag_kind', 'hist')
        height = kwargs.pop('height', 2.5)
        aspect = kwargs.pop('aspect', 1.0)
        plot_kws = kwargs.pop('plot_kws', {})
        diag_kws = kwargs.pop('diag_kws', {})
        markers = kwargs.pop('markers', 'o')

        n = len(numerical_cols)
        fig, axes = plt.subplots(n, n, figsize=(aspect * height * n, height * n))
        if n == 1:
            axes = np.array([[axes]])

        hue_arr = None
        hue_levels = None
        color_map = None
        if hue is not None and hue in df.columns:
            hue_arr = df[hue].to_numpy()
            hue_levels = np.unique(hue_arr)
            palette_colors = sns.color_palette(palette, n_colors=len(hue_levels))
            color_map = {level: palette_colors[idx] for idx, level in enumerate(hue_levels)}

        for i, y_col in enumerate(numerical_cols):
            y_arr = df[y_col].to_numpy().astype(float)
            for j, x_col in enumerate(numerical_cols):
                ax = axes[i, j]
                x_arr = df[x_col].to_numpy().astype(float)

                if i == j:
                    if hue_arr is None:
                        valid = ~np.isnan(x_arr)
                        if diag_kind == 'kde':
                            sns.kdeplot(x=x_arr[valid], ax=ax, **diag_kws)
                        else:
                            sns.histplot(x=x_arr[valid], ax=ax, **diag_kws)
                    else:
                        for level in hue_levels:
                            valid = (~np.isnan(x_arr)) & (hue_arr == level)
                            if np.any(valid):
                                if diag_kind == 'kde':
                                    sns.kdeplot(
                                        x=x_arr[valid],
                                        ax=ax,
                                        color=color_map[level],
                                        label=str(level),
                                        **diag_kws,
                                    )
                                else:
                                    sns.histplot(
                                        x=x_arr[valid],
                                        ax=ax,
                                        color=color_map[level],
                                        label=str(level),
                                        **diag_kws,
                                    )
                else:
                    if hue_arr is None:
                        valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr))
                        sns.scatterplot(
                            x=x_arr[valid],
                            y=y_arr[valid],
                            ax=ax,
                            marker=markers,
                            **plot_kws,
                        )
                    else:
                        for level in hue_levels:
                            valid = (~np.isnan(x_arr)) & (~np.isnan(y_arr)) & (hue_arr == level)
                            if np.any(valid):
                                sns.scatterplot(
                                    x=x_arr[valid],
                                    y=y_arr[valid],
                                    ax=ax,
                                    color=color_map[level],
                                    label=str(level),
                                    marker=markers,
                                    **plot_kws,
                                )

                if i < n - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel(x_col)

                if j > 0:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(y_col)

                # Avoid repeated legends across all cells.
                if hue_arr is not None and (i != 0 or j != n - 1):
                    legend = ax.get_legend()
                    if legend is not None:
                        legend.remove()

        # Keep one legend in the top-right subplot if hue is active.
        if hue_arr is not None:
            top_right_ax = axes[0, n - 1]
            handles, labels = top_right_ax.get_legend_handles_labels()
            if handles:
                top_right_ax.legend(handles, labels, title=hue)

        if title:
            fig.suptitle(title, y=1.02)

        plt.tight_layout()
        plt.show()


class PlotManager:
    """Facade for managing and standardizing EDA plots."""
    
    def __init__(self):
        self._hist_plotter = HistPlotter()
        self._box_plotter = BoxPlotter()
        self._scatter_plotter = ScatterPlotter()
        self._missingness_plotter = MissingnessPlotter()
        self._count_plotter = CountPlotter()
        self._correlation_plotter = CorrelationMatrixPlotter()
        self._pairplot_plotter = PairPlotMatrixPlotter()
        
    def histplot(self, df: DataFrame, column: str, bins: int = 30, title: str = None, **kwargs) -> None:
        """Plot a histogram for a column."""
        self._hist_plotter.plot(df, column=column, bins=bins, title=title, **kwargs)
        
    def boxplot(self, df: DataFrame, column: str, title: str = None, **kwargs) -> None:
        """Plot a boxplot for a column."""
        self._box_plotter.plot(df, column=column, title=title, **kwargs)
        
    def scatterplot(self, df: DataFrame, x: str, y: str, title: str = None, **kwargs) -> None:
        """Plot a scatterplot between two columns."""
        self._scatter_plotter.plot(df, x=x, y=y, title=title, **kwargs)
        
    def missingness(self, df: DataFrame, title: str = None, **kwargs) -> None:
        """Plot a missingness map for the entire DataFrame."""
        self._missingness_plotter.plot(df, title=title, **kwargs)
        
    def countplot(self, df: DataFrame, column: str, title: str = None, **kwargs) -> None:
        """Plot a countplot for categorical columns."""
        self._count_plotter.plot(df, column=column, title=title, **kwargs)
        
    def correlation_matrix(self, df: DataFrame, title: str = None, **kwargs) -> None:
        """Plot a correlation matrix heatmap."""
        self._correlation_plotter.plot(df, title=title, **kwargs)
        
    def pairplot(self, df: DataFrame, title: str = None, **kwargs) -> None:
        """Plot a pair plot matrix."""
        self._pairplot_plotter.plot(df, title=title, **kwargs)
        
    def multiplot(self, df: DataFrame, columns: list[str], plot_type: str = "hist", cols: int = 2, **kwargs) -> None:
        """Plot multiple plots in one panel.
        
        Enhanced to dynamically adapt the grid layout based on the number of plots,
        and allow custom layout arguments.
        """
        n = len(columns)
        if n == 0:
            return
            
        rows = (n + cols - 1) // cols
        
        # Calculate dynamic figure size
        figsize_x = kwargs.pop('figsize_x', cols * 5)
        figsize_y = kwargs.pop('figsize_y', rows * 4)
        
        fig, axes = plt.subplots(rows, cols, figsize=(figsize_x, figsize_y))
        
        if n == 1:
            axes_flat = [axes]
        else:
            axes_flat = np.array(axes).flatten()
        
        for i, col in enumerate(columns):
            ax = axes_flat[i]
            if plot_type == "hist":
                self._hist_plotter.plot(df, col, ax=ax, **kwargs)
            elif plot_type == "box":
                self._box_plotter.plot(df, col, ax=ax, **kwargs)
            elif plot_type == "count":
                self._count_plotter.plot(df, col, ax=ax, **kwargs)
            else:
                ax.set_title(f"Unsupported plot type: {plot_type}")
                
        # Hide empty subplots
        for j in range(len(columns), len(axes_flat)):
            axes_flat[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()

# Global instance for easy access
plot_manager = PlotManager()
