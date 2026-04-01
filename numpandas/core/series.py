"""Series implementation for numpandas."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .index import Index
from ..utils.dtypes import is_nan_value, safe_cast


class Series:
    """One-dimensional labeled array backed by a 2D NumPy array.

    Parameters
    ----------
    data : Iterable[Any] or numpy.ndarray
        Series data.
    index : Iterable[Any], optional
        Index labels. Defaults to ``range(len(data))``.
    name : str, optional
        Series name.
    """

    def __init__(self, data: Iterable[Any] | np.ndarray, index: Iterable[Any] | None = None, name: str | None = None):
        array = np.asarray(data)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != 1:
            raise ValueError("Series data must be 1D or a 2D column array.")
        self._data = array
        if index is None:
            index = range(array.shape[0])
        index_obj = Index(index)
        if len(index_obj) != array.shape[0]:
            raise ValueError("Index length does not match data length.")
        self._index = index_obj
        self._name = name

    @property
    def name(self) -> str | None:
        """Return series name."""
        return self._name

    @property
    def index(self) -> Index:
        """Return index object."""
        return self._index

    @property
    def shape(self) -> tuple[int, int]:
        """Return data shape."""
        return self._data.shape

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __str__(self) -> str:
        """Return a readable string representation."""
        lines = []
        values = self.to_numpy()
        for label, value in zip(self._index.to_list(), values):
            lines.append(f"{label}\t{value}")
        dtype_line = f"Dtype: {values.dtype}"
        if self._name is None:
            return "\n".join(lines + [dtype_line])
        return "\n".join(lines + [f"Name: {self._name}", dtype_line])

    def __repr__(self) -> str:
        """Return debug representation."""
        return self.__str__()

    def to_numpy(self) -> np.ndarray:
        """Return a 1D NumPy array of the series data.

        Returns
        -------
        numpy.ndarray
            Flattened data array.
        """
        return self._data.reshape(-1).copy()

    def to_list(self) -> list[Any]:
        """Return data as a list."""
        return self.to_numpy().tolist()

    def map(self, func: Callable[[Any], Any]) -> "Series":
        """Apply a function element-wise.

        Parameters
        ----------
        func : Callable[[Any], Any]
            Function to apply to each value.

        Returns
        -------
        Series
            New series with mapped values.
        """
        result = []
        for value in self.to_numpy():
            if is_nan_value(value):
                result.append(np.nan)
            else:
                result.append(func(value))
        return Series(result, index=self._index.to_list(), name=self._name)

    def isna(self) -> "Series":
        """Return boolean series indicating missing values.

        Returns
        -------
        Series
            Boolean series.
        """
        data = self.to_numpy()
        mask = np.array([is_nan_value(value) for value in data], dtype=bool)
        return Series(mask, index=self._index.to_list(), name=self._name)

    def fillna(self, value: Any) -> "Series":
        """Fill NaN values with a scalar.

        Parameters
        ----------
        value : Any
            Scalar to use for filling.

        Returns
        -------
        Series
            Filled series.
        """
        data = self.to_numpy()
        filled = np.array([value if is_nan_value(item) else item for item in data], dtype=object)
        return Series(filled, index=self._index.to_list(), name=self._name)

    def dropna(self) -> "Series":
        """Return a series with NaNs removed.

        Returns
        -------
        Series
            Series with NaNs removed.
        """
        data = self.to_numpy()
        mask = np.array([not is_nan_value(value) for value in data], dtype=bool)
        filtered = data[mask]
        new_index = [label for label, keep in zip(self._index.to_list(), mask) if keep]
        return Series(filtered, index=new_index, name=self._name)

    def astype(self, dtype: Any) -> "Series":
        """Cast series values to a dtype with safety checks.

        Parameters
        ----------
        dtype : Any
            Target dtype.

        Returns
        -------
        Series
            Casted series.
        """
        casted = safe_cast(self.to_numpy(), dtype)
        return Series(casted, index=self._index.to_list(), name=self._name)

    def count(self) -> int:
        """Return count of non-NaN values."""
        return int(np.sum(~self.isna().to_numpy()))

    def sum(self) -> Any:
        """Return sum of values, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nansum(data))

    def mean(self) -> float:
        """Return mean of values, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanmean(data))

    def std(self) -> float:
        """Return sample standard deviation, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanstd(data, ddof=1))

    def var(self) -> float:
        """Return sample variance, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanvar(data, ddof=1))

    def min(self) -> Any:
        """Return minimum value, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanmin(data))

    def max(self) -> Any:
        """Return maximum value, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanmax(data))

    def median(self) -> float:
        """Return median value, ignoring NaN."""
        data = self.to_numpy().astype(float, copy=False)
        return float(np.nanmedian(data))


class _SeriesLocIndexer:
    def __init__(self, series: Series):
        self._series = series

    def __getitem__(self, key):
        pos = self._series.index.get_loc(key)
        return self._series.to_numpy()[pos]


class _SeriesILocIndexer:
    def __init__(self, series: Series):
        self._series = series

    def __getitem__(self, key):
        return self._series.to_numpy()[key]


Series.loc = property(lambda self: _SeriesLocIndexer(self))
Series.iloc = property(lambda self: _SeriesILocIndexer(self))
