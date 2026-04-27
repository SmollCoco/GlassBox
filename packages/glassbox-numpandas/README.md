# glassbox-numpandas

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![NumPy-only Core](https://img.shields.io/badge/core-NumPy--only-2f9e44) ![Part of GlassBox](https://img.shields.io/badge/ecosystem-GlassBox-0b7285)

`glassbox-numpandas` is the NumPy-backed tabular foundation of GlassBox. It provides lightweight `DataFrame`, `Series`, and `Index` abstractions plus CSV/JSON/Excel I/O and dtype utilities for the rest of the ecosystem.

## Installation

```bash
pip install glassbox-numpandas
```

## Import Example

```python
from GlassBox.numpandas import DataFrame, read_csv
```

## Minimal Usage

```python
import numpy as np
from GlassBox.numpandas import DataFrame

df = DataFrame(
    {
        "age": np.array([20.0, 25.0, np.nan, 40.0]),
        "income": np.array([2000.0, 2500.0, 1800.0, 3200.0]),
    }
)

print(df.shape)
print(df.fillna({"age": 28.0}).mean().to_numpy())
```

## API Inventory

| Class | Purpose |
|---|---|
| `DataFrame` | 2D labeled table with columnar storage and tabular operations. |
| `_LocIndexer` | Internal label-based indexer for `DataFrame.loc`. |
| `_ILocIndexer` | Internal position-based indexer for `DataFrame.iloc`. |
| `Series` | 1D labeled array type. |
| `_SeriesLocIndexer` | Internal label-based indexer for `Series.loc`. |
| `_SeriesILocIndexer` | Internal position-based indexer for `Series.iloc`. |
| `Index` | Label container with lookup support. |
| `read_csv(filepath: str, sep: str = ",", header: bool = True) -> DataFrame` | Read CSV into a `DataFrame`. |
| `write_csv(frame: DataFrame, filepath: str, index: bool = False) -> None` | Write `DataFrame` to CSV. |
| `read_json(filepath: str) -> DataFrame` | Read JSON into a `DataFrame`. |
| `write_json(frame: DataFrame, filepath: str) -> None` | Write `DataFrame` to JSON. |
| `_to_json_scalar(value: Any) -> Any` | Internal JSON scalar normalization helper. |
| `read_excel(filepath: str, sheet_name: int | str = 0) -> DataFrame` | Read Excel sheet into a `DataFrame`. |
| `write_excel(frame: DataFrame, filepath: str, sheet_name: str = "Sheet1") -> None` | Write `DataFrame` to Excel. |
| `is_nan_value(value: Any) -> bool` | Detect NaN-like scalar values. |
| `array_has_nan(array: np.ndarray) -> bool` | Detect if an array contains NaN values. |
| `infer_dtype(values: Iterable[Any]) -> np.dtype` | Infer a NumPy dtype from parsed values. |
| `safe_cast(array: np.ndarray, dtype: Any) -> np.ndarray` | Cast arrays with safety checks. |
| `parse_scalar(value: Any) -> Any` | Parse scalar values from text-like sources. |
| `_normalize_list(indices, max_size: int)` | Internal helper for index list normalization. |
| `_slice_impl(frame: DataFrame, row_key, col_key)` | Internal slicing implementation for frame indexers. |
| `_rows_to_index(frame: DataFrame, rows)` | Internal helper to build row index labels. |
| `_selector_to_positions(selector, size: int)` | Internal selector normalization helper. |
| `_normalize_row_key(key, size: int)` | Internal row-key normalization helper. |
| `_normalize_col_key(key, size: int)` | Internal column-key normalization helper. |

## Repository

Main GlassBox GitHub repository: [https://github.com/SmollCoco/GlassBox](https://github.com/SmollCoco/GlassBox)
