"""DataFrame implementation for numpandas."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Iterable

import numpy as np

from .index import Index
from .series import Series
from ..utils.dtypes import is_nan_value, safe_cast


class DataFrame:
    """Two-dimensional labeled data structure with columnar storage.

    Parameters
    ----------
    data : dict[str, Iterable[Any]]
        Mapping of column name to data.
    columns : Iterable[str], optional
        Column order. Defaults to insertion order.
    index : Iterable[Any], optional
        Row labels. Defaults to ``range(n_rows)``.
    """

    def __init__(
        self,
        data: dict[str, Iterable[Any]] | None = None,
        columns: Iterable[str] | None = None,
        index: Iterable[Any] | None = None,
    ):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError("DataFrame data must be a dict of column arrays.")
        if columns is None:
            columns = list(data.keys())
        columns_list = list(columns)
        ordered = OrderedDict()
        lengths = []
        for column in columns_list:
            if column not in data:
                raise KeyError(f"Column not in data: {column}")
            col_array = np.asarray(data[column])
            if col_array.ndim == 2 and col_array.shape[1] == 1:
                col_array = col_array.reshape(-1)
            if col_array.ndim != 1:
                raise ValueError("Each column must be 1D or a 2D column array.")
            ordered[column] = col_array.copy()
            lengths.append(col_array.shape[0])
        if lengths:
            length = lengths[0]
            if any(item != length for item in lengths):
                raise ValueError("All columns must have the same length.")
        else:
            length = 0
        if index is None:
            index = range(length)
        index_obj = Index(index)
        if len(index_obj) != length:
            raise ValueError("Index length does not match data length.")
        self._data = ordered
        self._columns = columns_list
        self._index = index_obj

    @classmethod
    def from_numpy(cls, array: np.ndarray, columns: Iterable[str]):
        """Create a DataFrame from a NumPy array.

        Parameters
        ----------
        array : numpy.ndarray
            2D array of data.
        columns : Iterable[str]
            Column names.

        Returns
        -------
        DataFrame
            New DataFrame instance.
        """
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError("Input array must be 2D.")
        cols = list(columns)
        if array.shape[1] != len(cols):
            raise ValueError("Number of columns does not match array shape.")
        data = {col: array[:, idx] for idx, col in enumerate(cols)}
        return cls(data, columns=cols)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the DataFrame shape."""
        return (len(self._index), len(self._columns))

    @property
    def columns(self) -> list[str]:
        """Return list of column names."""
        return list(self._columns)

    @property
    def dtypes(self) -> dict[str, np.dtype]:
        """Return dict of column name to dtype."""
        return {name: self._data[name].dtype for name in self._columns}

    @property
    def index(self) -> Index:
        """Return index object."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __str__(self) -> str:
        """Return a readable string representation."""
        header = "index\t" + "\t".join(str(col) for col in self._columns)
        lines = [header]
        for row_idx, row_label in enumerate(self._index.to_list()):
            row = [str(row_label)] + [str(self._data[col][row_idx]) for col in self._columns]
            lines.append("\t".join(row))
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return debug representation."""
        return self.__str__()

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(f"Column not found: {key}")
            return Series(self._data[key], index=self._index.to_list(), name=key)
        if isinstance(key, (list, tuple)):
            subset = {col: self._data[col] for col in key}
            return DataFrame(subset, columns=list(key), index=self._index.to_list())
        if isinstance(key, Series):
            mask = np.asarray(key.to_numpy(), dtype=bool)
            return self._filter_rows(mask)
        mask_array = np.asarray(key)
        if mask_array.dtype == bool:
            return self._filter_rows(mask_array)
        raise TypeError("Invalid key type for DataFrame selection.")

    def _filter_rows(self, mask: np.ndarray) -> "DataFrame":
        if mask.shape[0] != len(self._index):
            raise ValueError("Boolean mask length does not match DataFrame.")
        data = {col: values[mask] for col, values in self._data.items()}
        new_index = [label for label, keep in zip(self._index.to_list(), mask) if keep]
        return DataFrame(data, columns=self._columns, index=new_index)

    def to_numpy(self) -> np.ndarray:
        """Return a 2D NumPy array of the DataFrame data."""
        if not self._columns:
            return np.empty((len(self._index), 0))
        return np.column_stack([self._data[col] for col in self._columns])

    def head(self, n: int = 5) -> "DataFrame":
        """Return first n rows."""
        return self.iloc[:n, :]

    def tail(self, n: int = 5) -> "DataFrame":
        """Return last n rows."""
        return self.iloc[-n:, :]

    def rename(self, columns: dict[str, str]) -> "DataFrame":
        """Return DataFrame with renamed columns.

        Parameters
        ----------
        columns : dict[str, str]
            Mapping of old name to new name.

        Returns
        -------
        DataFrame
            DataFrame with renamed columns.
        """
        new_columns = [columns.get(name, name) for name in self._columns]
        data = {new: self._data[old] for old, new in zip(self._columns, new_columns)}
        return DataFrame(data, columns=new_columns, index=self._index.to_list())

    def drop(self, columns: Iterable[str]) -> "DataFrame":
        """Return DataFrame with columns dropped.

        Parameters
        ----------
        columns : Iterable[str]
            Columns to drop.

        Returns
        -------
        DataFrame
            DataFrame without specified columns.
        """
        drop_set = set(columns)
        remaining = [name for name in self._columns if name not in drop_set]
        data = {name: self._data[name] for name in remaining}
        return DataFrame(data, columns=remaining, index=self._index.to_list())

    def reset_index(self) -> "DataFrame":
        """Reset index to a simple integer range."""
        return DataFrame(self._data, columns=self._columns, index=range(len(self._index)))

    def isna(self) -> "DataFrame":
        """Return DataFrame of booleans indicating missing values."""
        data = {}
        for name in self._columns:
            col = self._data[name]
            if np.issubdtype(col.dtype, np.floating):
                mask = np.isnan(col)
            elif col.dtype == object:
                mask = np.array([is_nan_value(value) for value in col], dtype=bool)
            else:
                mask = np.zeros(col.shape[0], dtype=bool)
            data[name] = mask
        return DataFrame(data, columns=self._columns, index=self._index.to_list())

    def fillna(self, value: Any | dict[str, Any]) -> "DataFrame":
        """Fill NaN values with a scalar or per-column dict.

        Parameters
        ----------
        value : Any or dict[str, Any]
            Scalar fill value or mapping per column.

        Returns
        -------
        DataFrame
            Filled DataFrame.
        """
        if isinstance(value, dict):
            unknown = set(value.keys()) - set(self._columns)
            if unknown:
                unknown_cols = ", ".join(sorted(str(item) for item in unknown))
                raise KeyError(f"Unknown column(s) in fillna mapping: {unknown_cols}")

        data = {}
        for name in self._columns:
            col = self._data[name]
            fill_value = value[name] if isinstance(value, dict) and name in value else value

            if isinstance(value, dict) and name not in value:
                data[name] = col.copy()
                continue

            if np.issubdtype(col.dtype, np.floating):
                mask = np.isnan(col)
                new_col = col.copy()
                new_col[mask] = fill_value
            elif col.dtype == object:
                new_col = np.array(
                    [fill_value if is_nan_value(val) else val for val in col],
                    dtype=object,
                )
            else:
                new_col = col.copy()
            data[name] = new_col
        return DataFrame(data, columns=self._columns, index=self._index.to_list())

    def dropna(self, axis: int = 0, how: str = "any") -> "DataFrame":
        """Drop rows or columns containing NaNs.

        Parameters
        ----------
        axis : int, default 0
            0 to drop rows, 1 to drop columns.
        how : {'any', 'all'}, default 'any'
            Whether to drop if any or all values are NaN.

        Returns
        -------
        DataFrame
            DataFrame with rows or columns removed.
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        if how not in ("any", "all"):
            raise ValueError("how must be 'any' or 'all'")
        isna_df = self.isna()
        if axis == 0:
            mask = np.zeros(len(self._index), dtype=bool)
            for name in self._columns:
                col_mask = isna_df._data[name]
                mask = np.logical_or(mask, col_mask)
            if how == "all":
                mask = np.ones(len(self._index), dtype=bool)
                for name in self._columns:
                    col_mask = isna_df._data[name]
                    mask = np.logical_and(mask, col_mask)
            keep = ~mask
            return self._filter_rows(keep)
        drop_cols = []
        for name in self._columns:
            col_mask = isna_df._data[name]
            if how == "any" and np.any(col_mask):
                drop_cols.append(name)
            if how == "all" and np.all(col_mask):
                drop_cols.append(name)
        remaining = [name for name in self._columns if name not in drop_cols]
        data = {name: self._data[name] for name in remaining}
        return DataFrame(data, columns=remaining, index=self._index.to_list())

    def astype(self, mapping: dict[str, Any]) -> "DataFrame":
        """Cast columns to specified dtypes with safety checks."""
        data = {}
        for name in self._columns:
            col = self._data[name]
            if name in mapping:
                data[name] = safe_cast(col, mapping[name])
            else:
                data[name] = col.copy()
        return DataFrame(data, columns=self._columns, index=self._index.to_list())

    def apply(self, func: Callable[[Any], Any], axis: int = 0):
        """Apply a function column-wise or row-wise.

        Parameters
        ----------
        func : Callable[[Any], Any]
            Function to apply.
        axis : int, default 0
            0 for columns, 1 for rows.

        Returns
        -------
        Series or DataFrame
            Result of applying function.
        """
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        if axis == 0:
            results = []
            for name in self._columns:
                series = Series(self._data[name], index=self._index.to_list(), name=name)
                results.append(func(series))
            if all(np.isscalar(item) or item is None for item in results):
                return Series(results, index=self._columns, name=None)
            lengths = []
            columns = []
            data = {}
            for name, result in zip(self._columns, results):
                arr = np.asarray(result)
                if arr.shape[0] != len(self._index):
                    raise ValueError("Apply result length does not match index.")
                data[name] = arr
                lengths.append(arr.shape[0])
                columns.append(name)
            return DataFrame(data, columns=columns, index=self._index.to_list())
        results = []
        for row_idx in range(len(self._index)):
            row = [self._data[col][row_idx] for col in self._columns]
            row_array = np.array(row, dtype=object)
            results.append(func(Series(row_array, index=self._columns)))
        if all(np.isscalar(item) or item is None for item in results):
            return Series(results, index=self._index.to_list(), name=None)
        data = {}
        for idx, col in enumerate(self._columns):
            data[col] = np.array([row[idx] for row in results])
        return DataFrame(data, columns=self._columns, index=self._index.to_list())

    def sample(self, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> "DataFrame":
        """Return a random sample of rows.

        Parameters
        ----------
        n : int, optional
            Number of rows.
        frac : float, optional
            Fraction of rows.
        random_state : int, optional
            Seed for RNG.

        Returns
        -------
        DataFrame
            Sampled DataFrame.
        """
        if (n is None) == (frac is None):
            raise ValueError("Specify exactly one of n or frac.")
        rng = np.random.default_rng(random_state)
        if frac is not None:
            if not (0 <= frac <= 1):
                raise ValueError("frac must be between 0 and 1.")
            n = int(round(len(self._index) * frac))
        if n is None:
            raise ValueError("n must be provided when frac is None.")
        if n < 0 or n > len(self._index):
            raise ValueError("n must be between 0 and the number of rows.")
        indices = rng.choice(len(self._index), size=n, replace=False)
        return self.iloc[indices]

    def count(self) -> Series:
        """Return count of non-NaN values per column."""
        counts = []
        for name in self._columns:
            col = self._data[name]
            if np.issubdtype(col.dtype, np.floating):
                counts.append(int(np.sum(~np.isnan(col))))
            elif col.dtype == object:
                counts.append(int(np.sum([not is_nan_value(val) for val in col])))
            else:
                counts.append(len(col))
        return Series(counts, index=self._columns, name=None)

    def sum(self) -> Series:
        """Return sum of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nansum(data.astype(float, copy=False))))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def mean(self) -> Series:
        """Return mean of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanmean(data.astype(float, copy=False))))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def std(self) -> Series:
        """Return sample standard deviation of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanstd(data.astype(float, copy=False), ddof=1)))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def var(self) -> Series:
        """Return sample variance of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanvar(data.astype(float, copy=False), ddof=1)))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def min(self) -> Series:
        """Return minimum of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanmin(data.astype(float, copy=False))))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def max(self) -> Series:
        """Return maximum of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanmax(data.astype(float, copy=False))))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def median(self) -> Series:
        """Return median of columns, ignoring NaN."""
        values = []
        for name in self._columns:
            data = self._data[name]
            if np.issubdtype(data.dtype, np.number):
                values.append(float(np.nanmedian(data.astype(float, copy=False))))
            else:
                raise TypeError(f"Column is non-numeric: {name}")
        return Series(values, index=self._columns, name=None)

    def describe(self) -> "DataFrame":
        """Return summary statistics for numeric columns."""
        stats = OrderedDict()
        labels = ["count", "mean", "std", "min", "median", "max", "var"]
        for name in self._columns:
            data = self._data[name]
            if not np.issubdtype(data.dtype, np.number):
                continue
            series = Series(data, index=self._index.to_list(), name=name)
            stats[name] = np.array(
                [
                    series.count(),
                    series.mean(),
                    series.std(),
                    series.min(),
                    series.median(),
                    series.max(),
                    series.var(),
                ],
                dtype=float,
            )
        return DataFrame(stats, columns=list(stats.keys()), index=labels)

    def to_csv(self, filepath: str, index: bool = False) -> None:
        """Write DataFrame to CSV file."""
        from ..io.csv import write_csv

        write_csv(self, filepath, index=index)

    def to_json(self, filepath: str) -> None:
        """Write DataFrame to JSON file."""
        from ..io.json import write_json

        write_json(self, filepath)

    def to_excel(self, filepath: str, sheet_name: str = "Sheet1") -> None:
        """Write DataFrame to Excel file."""
        from ..io.excel import write_excel

        write_excel(self, filepath, sheet_name=sheet_name)


class _LocIndexer:
    def __init__(self, frame: DataFrame):
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        row_sel = self._resolve_row_labels(row_key)
        col_sel = self._resolve_col_labels(col_key)
        return self._frame._slice(row_sel, col_sel)

    def _resolve_row_labels(self, key):
        if isinstance(key, slice):
            start = self._frame.index.get_loc(key.start) if key.start is not None else None
            stop = self._frame.index.get_loc(key.stop) + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
        if isinstance(key, list):
            return [self._frame.index.get_loc(item) for item in key]
        return self._frame.index.get_loc(key)

    def _resolve_col_labels(self, key):
        if isinstance(key, slice):
            cols = self._frame.columns
            start = cols.index(key.start) if key.start is not None else None
            stop = cols.index(key.stop) + 1 if key.stop is not None else None
            return slice(start, stop, key.step)
        if isinstance(key, list):
            return [self._frame.columns.index(item) for item in key]
        return self._frame.columns.index(key)


class _ILocIndexer:
    def __init__(self, frame: DataFrame):
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        return self._frame._slice(row_key, col_key)


DataFrame.loc = property(lambda self: _LocIndexer(self))
DataFrame.iloc = property(lambda self: _ILocIndexer(self))


def _normalize_list(indices, max_size: int):
    return [idx + max_size if idx < 0 else idx for idx in indices]


DataFrame._slice = lambda self, row_key, col_key: _slice_impl(self, row_key, col_key)


def _slice_impl(frame: DataFrame, row_key, col_key):
    rows = _normalize_row_key(row_key, len(frame.index))
    cols = _normalize_col_key(col_key, len(frame.columns))

    row_positions = _selector_to_positions(rows, len(frame.index))
    col_positions = _selector_to_positions(cols, len(frame.columns))

    if np.isscalar(row_positions) and np.isscalar(col_positions):
        return frame._data[frame.columns[col_positions]][row_positions]

    if np.isscalar(row_positions):
        col_labels = [frame.columns[col] for col in col_positions]
        values = [frame._data[col][row_positions] for col in col_labels]
        return Series(values, index=col_labels, name=frame.index.to_list()[row_positions])

    if np.isscalar(col_positions):
        col_name = frame.columns[col_positions]
        values = frame._data[col_name][row_positions]
        new_index = _rows_to_index(frame, row_positions)
        return Series(values, index=new_index, name=col_name)

    col_labels = [frame.columns[col] for col in col_positions]
    data = {col: frame._data[col][row_positions] for col in col_labels}
    new_index = _rows_to_index(frame, row_positions)
    return DataFrame(data, columns=col_labels, index=new_index)


def _rows_to_index(frame: DataFrame, rows):
    if isinstance(rows, list):
        return [frame.index.to_list()[i] for i in rows]
    return [frame.index.to_list()[i] for i in range(*rows.indices(len(frame.index)))]


def _selector_to_positions(selector, size: int):
    if isinstance(selector, slice):
        return list(range(*selector.indices(size)))
    if isinstance(selector, list):
        return selector
    return selector


def _normalize_row_key(key, size: int):
    if isinstance(key, slice):
        return key
    if isinstance(key, list) or isinstance(key, np.ndarray):
        return _normalize_list(list(key), size)
    if isinstance(key, (int, np.integer)):
        return int(key)
    raise TypeError("Invalid row selector for iloc/loc.")


def _normalize_col_key(key, size: int):
    if isinstance(key, slice):
        return key
    if isinstance(key, list) or isinstance(key, np.ndarray):
        return _normalize_list(list(key), size)
    if isinstance(key, (int, np.integer)):
        return int(key)
    raise TypeError("Invalid column selector for iloc/loc.")
