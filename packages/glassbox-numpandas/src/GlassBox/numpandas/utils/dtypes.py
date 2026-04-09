"""dtype inference and safe casting utilities."""

from __future__ import annotations

from typing import Iterable, Any

import numpy as np


def is_nan_value(value: Any) -> bool:
    """Return True if the value is a NaN-like scalar.

    Parameters
    ----------
    value : Any
        Scalar value to test.

    Returns
    -------
    bool
        True if the value is NaN, False otherwise.
    """
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def array_has_nan(array: np.ndarray) -> bool:
    """Return True if the array contains NaN values.

    Parameters
    ----------
    array : numpy.ndarray
        Array to inspect.

    Returns
    -------
    bool
        True if any NaN is present, False otherwise.
    """
    if np.issubdtype(array.dtype, np.floating):
        return bool(np.isnan(array).any())
    if array.dtype == object:
        return any(is_nan_value(value) for value in array)
    return False


def infer_dtype(values: Iterable[Any]) -> np.dtype:
    """Infer a NumPy dtype from a sequence of parsed values.

    Parameters
    ----------
    values : Iterable[Any]
        Parsed values (ints, floats, strings, NaN).

    Returns
    -------
    numpy.dtype
        Inferred dtype for the values.
    """
    has_str = False
    has_float = False
    has_int = False
    for value in values:
        if is_nan_value(value):
            has_float = True
            continue
        if isinstance(value, str):
            has_str = True
            continue
        if isinstance(value, (float, np.floating)):
            has_float = True
            continue
        if isinstance(value, (int, np.integer)):
            has_int = True
            continue
        has_str = True

    if has_str:
        return np.dtype("O")
    if has_float:
        return np.dtype("float64")
    if has_int:
        return np.dtype("int64")
    return np.dtype("float64")


def safe_cast(array: np.ndarray, dtype: Any) -> np.ndarray:
    """Cast an array to dtype while rejecting unsafe conversions.

    Parameters
    ----------
    array : numpy.ndarray
        Input array.
    dtype : Any
        Target dtype.

    Returns
    -------
    numpy.ndarray
        Cast array.

    Raises
    ------
    ValueError
        If the cast would drop information (e.g., NaN to int).
    TypeError
        If the cast is unsupported or unsafe.
    """
    target = np.dtype(dtype)
    if np.issubdtype(target, np.integer):
        if array_has_nan(array):
            raise ValueError("Cannot cast to integer with NaN present.")
        if np.issubdtype(array.dtype, np.floating):
            if not np.all(np.equal(array, np.floor(array))):
                raise ValueError("Cannot cast non-integer floats to integer.")
        return array.astype(target)
    if np.issubdtype(target, np.floating):
        return array.astype(target)
    if np.issubdtype(target, np.bool_):
        if array_has_nan(array):
            raise ValueError("Cannot cast to bool with NaN present.")
        return array.astype(target)
    if target.kind in ("U", "S", "O"):
        return array.astype(target)
    if not np.can_cast(array.dtype, target, casting="safe"):
        raise TypeError(f"Unsafe cast from {array.dtype} to {target}.")
    return array.astype(target)


def parse_scalar(value: Any) -> Any:
    """Parse a scalar from CSV/Excel/JSON into a typed value.

    Parameters
    ----------
    value : Any
        Scalar input.

    Returns
    -------
    Any
        Parsed scalar (int, float, str, or np.nan).
    """
    if value is None:
        return np.nan
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return np.nan
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    return value
