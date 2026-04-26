from __future__ import annotations

from GlassBox.benchmark.compare import BenchmarkResult, showcase_cases


def runtime(group: str, case: str, library: str, seconds: float) -> BenchmarkResult:
    return BenchmarkResult(group, case, library, "runtime_mean", seconds, "seconds")


def test_showcase_keeps_glassbox_wins_and_two_baseline_wins():
    results = [
        runtime("Metrics", "accuracy", "GlassBox", 0.1),
        runtime("Metrics", "accuracy", "pandas/sklearn", 1.0),
        runtime("DataFrame", "sum", "GlassBox", 0.2),
        runtime("DataFrame", "sum", "pandas/sklearn", 0.4),
        runtime("Models", "linear regression", "GlassBox", 1.0),
        runtime("Models", "linear regression", "pandas/sklearn", 0.1),
        runtime("Models", "logistic regression", "GlassBox", 1.0),
        runtime("Models", "logistic regression", "pandas/sklearn", 0.2),
        runtime("Preprocessing", "scaler", "GlassBox", 1.0),
        runtime("Preprocessing", "scaler", "pandas/sklearn", 0.3),
    ]

    cases = showcase_cases(results)
    winners = [item[3] for item in cases]

    assert winners.count("GlassBox") == 2
    assert winners.count("pandas/sklearn") == 2
    assert cases[0][1] == "accuracy"


def test_showcase_orders_glassbox_wins_by_speedup():
    results = [
        runtime("A", "small win", "GlassBox", 0.5),
        runtime("A", "small win", "pandas/sklearn", 1.0),
        runtime("A", "large win", "GlassBox", 0.1),
        runtime("A", "large win", "pandas/sklearn", 1.0),
        runtime("B", "baseline win", "GlassBox", 1.0),
        runtime("B", "baseline win", "pandas/sklearn", 0.1),
    ]

    cases = showcase_cases(results)

    assert cases[0][1] == "large win"
    assert cases[1][1] == "small win"
    assert cases[2][3] == "pandas/sklearn"
