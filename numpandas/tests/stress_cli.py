"""Command-line stress tests for numpandas.

Run:
    python numpandas/tests/stress_cli.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import numpandas as npd


def run_test(name, fn):
    try:
        fn()
        print(f"PASS - {name}")
    except Exception as exc:
        print(f"FAIL - {name}: {type(exc).__name__}: {exc}")


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


def test_cow_correctness():
    df = npd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})
    derived_df = df.fillna(0)
    derived_df._data["a"][0] = 999.0
    assert df._data["a"][0] == 1.0

    series = df["a"]
    derived_series = series.fillna(0)
    derived_series._data[0, 0] = -5
    assert df._data["a"][0] == 1.0


def test_loc_iloc_edge_indices():
    df = npd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}, index=["r0", "r1", "r2"])

    assert df.iloc[-1, 0] == 3
    assert df.iloc[[0, -1]].shape == (2, 2)
    assert df.loc["r1", "b"] == 20

    row = df.iloc[1]
    assert isinstance(row, npd.Series)
    assert row.to_list() == [2, 20]

    subset = df.iloc[[0, 2], [1]]
    assert isinstance(subset, npd.DataFrame)
    assert subset.shape == (2, 1)

    assert_raises(IndexError, lambda: df.iloc[10, 0])
    assert_raises(IndexError, lambda: df.iloc[0, 10])
    assert_raises(KeyError, lambda: df.loc["missing", "a"])


def test_dropna_all_axis_how_combinations():
    df = npd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan],
            "b": [np.nan, 2.0, np.nan],
            "c": ["x", "y", "z"],
            "d": [np.nan, np.nan, np.nan],
        }
    )

    out_any_row = df.dropna(axis=0, how="any")
    assert out_any_row.shape == (0, 4)

    out_all_row = df.dropna(axis=0, how="all")
    assert out_all_row.shape == (3, 4)

    out_any_col = df.dropna(axis=1, how="any")
    assert out_any_col.columns == ["c"]

    out_all_col = df.dropna(axis=1, how="all")
    assert out_all_col.columns == ["a", "b", "c"]


def test_astype_nan_to_int_raises():
    df = npd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert_raises(ValueError, df.astype, {"a": "int64"})


def test_fillna_with_nonexistent_dict_keys():
    df = npd.DataFrame({"a": [1.0, np.nan], "b": [2.0, np.nan]})
    assert_raises(KeyError, df.fillna, {"a": 0, "x": 10})


def test_apply_axis_shapes():
    df = npd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [10, 20, 30], "c": ["x", "y", "z"]})

    axis0 = df.apply(lambda s: s.count(), axis=0)
    assert isinstance(axis0, npd.Series)
    assert axis0.shape == (3, 1)

    axis1 = df.apply(lambda r: r.iloc[0] + r.iloc[1], axis=1)
    assert isinstance(axis1, npd.Series)
    vals = axis1.to_list()
    assert vals[0] == 11.0 and vals[1] == 22.0 and np.isnan(vals[2])


def test_sample_variants():
    df = npd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

    full = df.sample(frac=1.0, random_state=123)
    assert full.shape == df.shape
    assert sorted(full["a"].to_list()) == sorted(df["a"].to_list())

    s1 = df.sample(n=2, random_state=99)
    s2 = df.sample(n=2, random_state=99)
    assert np.array_equal(s1.to_numpy(), s2.to_numpy())

    assert_raises(ValueError, df.sample, n=10)


def test_io_roundtrips_with_nan():
    df = npd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [10, 20, 30]})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        csv_path = root / "x.csv"
        df.to_csv(str(csv_path), index=False)
        csv_back = npd.read_csv(str(csv_path))
        assert np.isnan(csv_back["a"].to_list()[1])
        assert csv_back["a"].to_list()[1] != "nan"
        assert csv_back["a"].to_list()[1] != ""

        json_path = root / "x.json"
        df.to_json(str(json_path))
        json_back = npd.read_json(str(json_path))
        assert np.isnan(json_back["a"].to_list()[1])
        assert json_back["a"].to_list()[1] != "nan"
        assert json_back["a"].to_list()[1] != ""

        try:
            excel_path = root / "x.xlsx"
            df.to_excel(str(excel_path))
            excel_back = npd.read_excel(str(excel_path))
            assert np.isnan(excel_back["a"].to_list()[1])
            assert excel_back["a"].to_list()[1] != "nan"
            assert excel_back["a"].to_list()[1] != ""
        except ImportError as exc:
            raise AssertionError(f"Excel test requires openpyxl: {exc}") from exc


def test_boolean_mask_length_mismatch_raises():
    df = npd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    mask = npd.Series([True, False], index=[0, 1])
    assert_raises(ValueError, lambda: df[mask])


def test_describe_with_fully_nan_column():
    df = npd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan, np.nan], "c": ["x", "y", "z"]})
    desc = df.describe()
    assert isinstance(desc, npd.DataFrame)
    assert desc.columns == ["a", "b"]
    mean_b = desc.loc["mean", "b"]
    assert np.isnan(mean_b)


def main():
    tests = [
        ("CoW correctness", test_cow_correctness),
        ("loc/iloc edge indices", test_loc_iloc_edge_indices),
        ("dropna axis/how combinations", test_dropna_all_axis_how_combinations),
        ("astype NaN->int raises", test_astype_nan_to_int_raises),
        ("fillna dict with unknown keys raises", test_fillna_with_nonexistent_dict_keys),
        ("apply axis return shapes", test_apply_axis_shapes),
        ("sample frac/n/random_state cases", test_sample_variants),
        ("CSV/JSON/Excel NaN round-trips", test_io_roundtrips_with_nan),
        ("boolean mask length mismatch", test_boolean_mask_length_mismatch_raises),
        ("describe with fully-NaN column", test_describe_with_fully_nan_column),
    ]

    for name, fn in tests:
        run_test(name, fn)


if __name__ == "__main__":
    main()
