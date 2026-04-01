# `numpandas` Developer Manual

## 1) Overview
- `numpandas` is a minimal pandas-like tabular library built with `Python + NumPy`.
- Scope is intentionally narrow for ML/data pipeline needs (GlassBox-oriented).
- It is **not** a full pandas replacement.
- Not implemented: `groupby`, joins/merges, pivot/melt/stack, MultiIndex, datetime/timeseries APIs.
- Dependencies:
  - Required: `numpy`
  - Optional for Excel I/O: `openpyxl`

## 2) Core Concepts
- **Copy-on-Write (CoW):** operations return new objects; source objects are not mutated.
- **Unified missing value:** use `np.nan` consistently.
- **No `inplace`:** no public API supports in-place mutation.
- **Columnar layout:** `DataFrame` stores columns independently (`column -> ndarray`).

## 3) Data Structures
| Type | Purpose | Construction | Key properties |
|---|---|---|---|
| `Index` | Label container | `Index(labels)` | label lookup via `get_loc` |
| `Series` | 1D labeled data | `Series(data, index=None, name=None)` | `name`, `index`, `shape`, `to_numpy`, `to_list` |
| `DataFrame` | 2D tabular data | `DataFrame(dict_of_columns, columns=None, index=None)` | `shape`, `columns`, `dtypes`, `index`, `to_numpy` |

## 4) Selection & Indexing
- `df[col]` → returns a `Series`.
- `df[[col1, col2]]` → returns a `DataFrame` with selected columns.
- `df.loc[row_label, col_label]` → label-based selection.
- `df.iloc[row_pos, col_pos]` → integer-position selection.
- `df[bool_mask]` accepts boolean `Series`/array with same row length; mismatched length raises.

## 5) Missing Values
- `df.isna()` / `series.isna()` returns boolean structure of same shape.
- `df.fillna(value)`:
  - Scalar: fill all NaNs.
  - Dict: fill only mapped columns.
  - Unknown dict keys raise `KeyError`.
- `df.dropna(axis=0|1, how='any'|'all')`:
  - `axis=0`: drop rows.
  - `axis=1`: drop columns.
  - `how='any'`: drop when any NaN present.
  - `how='all'`: drop when all values are NaN.

## 6) Type Casting
- `df.astype({col: dtype})` casts selected columns.
- `series.astype(dtype)` casts a series.
- Unsafe casts raise clearly:
  - NaN → integer raises `ValueError`.
  - Other unsafe conversions raise `TypeError` or `ValueError` depending on case.

## 7) Apply & Map
- `df.apply(func, axis=0)`:
  - Applies to each column as `Series`.
  - Scalar outputs return `Series` indexed by columns.
- `df.apply(func, axis=1)`:
  - Applies to each row as `Series`.
  - Scalar outputs return `Series` indexed by row index.
- `series.map(func)` applies element-wise; NaN values remain NaN.

## 8) Stats & Describe
- `Series` stats: `count`, `sum`, `mean`, `std`, `var`, `min`, `max`, `median`.
- `DataFrame` stats: same methods column-wise on numeric columns.
- `df.describe()` returns summary `DataFrame` with index:
  - `count`, `mean`, `std`, `min`, `median`, `max`, `var`
- Fully-NaN numeric columns can produce `NaN` statistics and NumPy runtime warnings.

## 9) Sampling & Slicing
- `df.sample(n=..., random_state=...)` random row sample without replacement.
- `df.sample(frac=...)` samples fraction of rows (`0 <= frac <= 1`).
- Exactly one of `n` or `frac` must be provided.
- `head(n)` returns first `n` rows as `DataFrame`.
- `tail(n)` returns last `n` rows as `DataFrame`.
- `iloc` supports scalar, list, and slice selectors.

## 10) I/O
- Read:
  - `read_csv(path, sep=',', header=True)`
  - `read_json(path)`
  - `read_excel(path, sheet_name=0)`
- Write:
  - `df.to_csv(path, index=False)`
  - `df.to_json(path)`
  - `df.to_excel(path, sheet_name='Sheet1')`
- NaN behavior:
  - Missing values load as `np.nan`.
  - JSON writer converts NaN to JSON `null` and restores as `np.nan` on read.

## 11) Utilities
- Structural: `shape`, `columns`, `dtypes`, `index`.
- Subsetting helpers: `head`, `tail`.
- Schema helpers: `rename(columns={...})`, `drop(columns=[...])`.
- Index helper: `reset_index()` resets to positional integer index.
- NumPy bridge:
  - `DataFrame.from_numpy(array, columns)`
  - `df.to_numpy()`

## 12) Error Conventions
- `TypeError` is used for invalid input types or unsupported selector types.
- `ValueError` is used for invalid values/shape mismatches (e.g., mask length mismatch, invalid sampling args, unsafe numeric cast cases).
- `KeyError` is used for missing labels/columns or unknown mapping keys.
- Philosophy: fail early with descriptive exceptions; no silent coercion for unsafe operations.
