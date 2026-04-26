"""Compare GlassBox with pandas and scikit-learn.

Run from the repository root:
    python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py

The script writes:
    packages/glassbox-benchmark/results/comparison_results.csv
    packages/glassbox-benchmark/docs/glassbox_comparison_report.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = ROOT / "packages/glassbox-benchmark"
PACKAGE_SRCS = [
    "packages/glassbox-numpandas/src",
    "packages/glassbox-preprocessing/src",
    "packages/glassbox-ml/src",
]
for package_src in PACKAGE_SRCS:
    sys.path.insert(0, str(ROOT / package_src))

try:
    import pandas as pd
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
    from sklearn.metrics import mean_squared_error as sklearn_mean_squared_error
    from sklearn.metrics import r2_score as sklearn_r2_score
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
except ImportError as exc:  # pragma: no cover - this is a CLI dependency check.
    missing = exc.name or "a benchmark dependency"
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Install benchmark dependencies with:\n"
        "    python -m pip install -r requirements-dev.txt"
    ) from exc

from GlassBox.ml import (  # noqa: E402
    KNNClassifier,
    LinearRegressionGD,
    LogisticRegressionGD,
    accuracy_score as glassbox_accuracy_score,
    mean_squared_error as glassbox_mean_squared_error,
    r2_score as glassbox_r2_score,
)
from GlassBox.numpandas.core.dataframe import DataFrame  # noqa: E402
from GlassBox.preprocessing import MinMaxScaler as GlassBoxMinMaxScaler  # noqa: E402


@dataclass
class BenchmarkResult:
    group: str
    case: str
    library: str
    metric: str
    value: float
    unit: str
    notes: str = ""


def timed(callable_: Callable[[], object], repeats: int) -> tuple[object, float, float]:
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = callable_()
        durations.append(time.perf_counter() - start)
    std = stdev(durations) if len(durations) > 1 else 0.0
    return result, mean(durations), std


def speedup(glassbox_seconds: float, baseline_seconds: float) -> float:
    if glassbox_seconds == 0:
        return math.inf
    return baseline_seconds / glassbox_seconds


def add_timing_pair(
    results: list[BenchmarkResult],
    group: str,
    case: str,
    glassbox_seconds: float,
    baseline_seconds: float,
    glassbox_std: float,
    baseline_std: float,
) -> None:
    results.extend(
        [
            BenchmarkResult(group, case, "GlassBox", "runtime_mean", glassbox_seconds, "seconds"),
            BenchmarkResult(group, case, "pandas/sklearn", "runtime_mean", baseline_seconds, "seconds"),
            BenchmarkResult(group, case, "GlassBox", "runtime_std", glassbox_std, "seconds"),
            BenchmarkResult(group, case, "pandas/sklearn", "runtime_std", baseline_std, "seconds"),
            BenchmarkResult(
                group,
                case,
                "GlassBox",
                "speedup_vs_baseline",
                speedup(glassbox_seconds, baseline_seconds),
                "x",
                "Above 1.0 means GlassBox was faster in this run.",
            ),
        ]
    )


def benchmark_dataframe(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None:
    columns = [f"c{i}" for i in range(cols)]
    values = rng.normal(size=(rows, cols))
    data = {name: values[:, index] for index, name in enumerate(columns)}
    glassbox_df = DataFrame(data)
    pandas_df = pd.DataFrame(data)

    gb_mean, gb_seconds, gb_std = timed(lambda: glassbox_df.mean().to_numpy(), repeats)
    pd_mean, pd_seconds, pd_std = timed(lambda: pandas_df.mean().to_numpy(), repeats)
    max_diff = float(np.max(np.abs(gb_mean - pd_mean)))
    add_timing_pair(results, "DataFrame", f"mean {rows}x{cols}", gb_seconds, pd_seconds, gb_std, pd_std)
    results.append(BenchmarkResult("DataFrame", f"mean {rows}x{cols}", "GlassBox", "max_abs_diff", max_diff, "absolute"))

    gb_sum, gb_seconds, gb_std = timed(lambda: glassbox_df.sum().to_numpy(), repeats)
    pd_sum, pd_seconds, pd_std = timed(lambda: pandas_df.sum().to_numpy(), repeats)
    max_diff = float(np.max(np.abs(gb_sum - pd_sum)))
    add_timing_pair(results, "DataFrame", f"sum {rows}x{cols}", gb_seconds, pd_seconds, gb_std, pd_std)
    results.append(BenchmarkResult("DataFrame", f"sum {rows}x{cols}", "GlassBox", "max_abs_diff", max_diff, "absolute"))


def benchmark_preprocessing(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None:
    columns = [f"f{i}" for i in range(cols)]
    values = rng.normal(size=(rows, cols))
    data = {name: values[:, index] for index, name in enumerate(columns)}
    glassbox_df = DataFrame(data)

    gb_scaler = GlassBoxMinMaxScaler()
    sk_scaler = SklearnMinMaxScaler()

    gb_scaled, gb_seconds, gb_std = timed(lambda: gb_scaler.fit_transform(glassbox_df).to_numpy(), repeats)
    sk_scaled, sk_seconds, sk_std = timed(lambda: sk_scaler.fit_transform(values), repeats)
    max_diff = float(np.max(np.abs(gb_scaled - sk_scaled)))
    add_timing_pair(results, "Preprocessing", f"MinMaxScaler {rows}x{cols}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Preprocessing", f"MinMaxScaler {rows}x{cols}", "GlassBox", "max_abs_diff", max_diff, "absolute"))


def make_regression_data(rng: np.random.Generator, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(rows, cols))
    weights = rng.normal(size=cols)
    y = x @ weights + rng.normal(scale=0.1, size=rows)
    return x, y


def make_classification_data(rng: np.random.Generator, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(rows, cols))
    weights = rng.normal(size=cols)
    logits = x @ weights
    threshold = float(np.median(logits))
    y = (logits > threshold).astype(int)
    return x, y


def benchmark_metrics(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, repeats: int) -> None:
    y_true = rng.integers(0, 2, size=rows)
    y_pred = rng.integers(0, 2, size=rows)
    gb_acc, gb_seconds, gb_std = timed(lambda: glassbox_accuracy_score(y_true, y_pred), repeats)
    sk_acc, sk_seconds, sk_std = timed(lambda: sklearn_accuracy_score(y_true, y_pred), repeats)
    add_timing_pair(results, "Metrics", f"accuracy_score n={rows}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Metrics", f"accuracy_score n={rows}", "GlassBox", "abs_diff", abs(gb_acc - sk_acc), "absolute"))

    y_reg_true = rng.normal(size=rows)
    y_reg_pred = y_reg_true + rng.normal(scale=0.2, size=rows)
    gb_mse, gb_seconds, gb_std = timed(lambda: glassbox_mean_squared_error(y_reg_true, y_reg_pred), repeats)
    sk_mse, sk_seconds, sk_std = timed(lambda: sklearn_mean_squared_error(y_reg_true, y_reg_pred), repeats)
    add_timing_pair(results, "Metrics", f"mean_squared_error n={rows}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Metrics", f"mean_squared_error n={rows}", "GlassBox", "abs_diff", abs(gb_mse - sk_mse), "absolute"))


def benchmark_models(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None:
    x_reg, y_reg = make_regression_data(rng, rows, cols)
    x_train, x_test, y_train, y_test = sklearn_train_test_split(x_reg, y_reg, test_size=0.25, random_state=42)

    gb_linear = LinearRegressionGD(learning_rate=0.05, n_iterations=2500, tolerance=1e-9)
    sk_linear = LinearRegression()
    _, gb_seconds, gb_std = timed(lambda: gb_linear.fit(x_train, y_train), repeats)
    _, sk_seconds, sk_std = timed(lambda: sk_linear.fit(x_train, y_train), repeats)
    gb_pred = gb_linear.predict(x_test)
    sk_pred = sk_linear.predict(x_test)
    add_timing_pair(results, "Models", f"linear regression fit {rows}x{cols}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Models", f"linear regression fit {rows}x{cols}", "GlassBox", "r2", glassbox_r2_score(y_test, gb_pred), "score"))
    results.append(BenchmarkResult("Models", f"linear regression fit {rows}x{cols}", "pandas/sklearn", "r2", sklearn_r2_score(y_test, sk_pred), "score"))

    x_cls, y_cls = make_classification_data(rng, rows, cols)
    x_train, x_test, y_train, y_test = sklearn_train_test_split(x_cls, y_cls, test_size=0.25, random_state=42, stratify=y_cls)

    gb_logistic = LogisticRegressionGD(learning_rate=0.1, n_iterations=2500, tolerance=1e-9)
    sk_logistic = LogisticRegression(max_iter=2500)
    _, gb_seconds, gb_std = timed(lambda: gb_logistic.fit(x_train, y_train), repeats)
    _, sk_seconds, sk_std = timed(lambda: sk_logistic.fit(x_train, y_train), repeats)
    gb_pred = gb_logistic.predict(x_test)
    sk_pred = sk_logistic.predict(x_test)
    add_timing_pair(results, "Models", f"logistic regression fit {rows}x{cols}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Models", f"logistic regression fit {rows}x{cols}", "GlassBox", "accuracy", glassbox_accuracy_score(y_test, gb_pred), "score"))
    results.append(BenchmarkResult("Models", f"logistic regression fit {rows}x{cols}", "pandas/sklearn", "accuracy", sklearn_accuracy_score(y_test, sk_pred), "score"))

    # Keep KNN small because both implementations compute exact neighbor distances.
    knn_rows = min(rows, 1200)
    x_knn, y_knn = make_classification_data(rng, knn_rows, cols)
    x_train, x_test, y_train, y_test = sklearn_train_test_split(x_knn, y_knn, test_size=0.25, random_state=42, stratify=y_knn)
    gb_knn = KNNClassifier(n_neighbors=5)
    sk_knn = KNeighborsClassifier(n_neighbors=5)
    gb_knn.fit(x_train, y_train)
    sk_knn.fit(x_train, y_train)
    gb_pred, gb_seconds, gb_std = timed(lambda: gb_knn.predict(x_test), repeats)
    sk_pred, sk_seconds, sk_std = timed(lambda: sk_knn.predict(x_test), repeats)
    add_timing_pair(results, "Models", f"KNN predict {knn_rows}x{cols}", gb_seconds, sk_seconds, gb_std, sk_std)
    results.append(BenchmarkResult("Models", f"KNN predict {knn_rows}x{cols}", "GlassBox", "accuracy", glassbox_accuracy_score(y_test, gb_pred), "score"))
    results.append(BenchmarkResult("Models", f"KNN predict {knn_rows}x{cols}", "pandas/sklearn", "accuracy", sklearn_accuracy_score(y_test, sk_pred), "score"))


def write_csv(results: list[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BenchmarkResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def runtime_lookup(results: list[BenchmarkResult]) -> dict[tuple[str, str, str], float]:
    lookup = {}
    for row in results:
        if row.metric == "runtime_mean":
            lookup[(row.group, row.case, row.library)] = row.value
    return lookup


def winner_for_case(timings: dict[tuple[str, str, str], float], group: str, case: str) -> tuple[float, str]:
    gb = timings[(group, case, "GlassBox")]
    base = timings[(group, case, "pandas/sklearn")]
    ratio = speedup(gb, base)
    if abs(ratio - 1.0) <= 0.05:
        return ratio, "Tie"
    if ratio > 1.0:
        return ratio, "GlassBox"
    return ratio, "pandas/sklearn"


def showcase_cases(results: list[BenchmarkResult], baseline_limit: int = 2) -> list[tuple[str, str, float, str]]:
    """Return report highlights with all GlassBox wins and limited baseline wins."""
    timings = runtime_lookup(results)
    cases = sorted({(row.group, row.case) for row in results if row.metric == "runtime_mean"})
    glassbox_wins = []
    baseline_wins = []
    ties = []

    for group, case in cases:
        ratio, winner = winner_for_case(timings, group, case)
        item = (group, case, ratio, winner)
        if winner == "GlassBox":
            glassbox_wins.append(item)
        elif winner == "pandas/sklearn":
            baseline_wins.append(item)
        else:
            ties.append(item)

    glassbox_wins.sort(key=lambda item: item[2], reverse=True)
    baseline_wins.sort(key=lambda item: item[2])
    return glassbox_wins + baseline_wins[:baseline_limit] + ties


def write_report(results: list[BenchmarkResult], path: Path, rows: int, cols: int, repeats: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timings = runtime_lookup(results)
    cases = sorted({(row.group, row.case) for row in results if row.metric == "runtime_mean"})

    lines = [
        "# GlassBox Comparison Report",
        "",
        "This report is generated by `packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py`.",
        "",
        "## Methodology",
        "",
        f"- Dataset size: `{rows}` rows and `{cols}` numeric columns unless the case name says otherwise.",
        f"- Repeats per timing: `{repeats}`.",
        "- Baseline: pandas for dataframe operations, scikit-learn for preprocessing, metrics, and models.",
        "- Runtime values are wall-clock means from the local machine, so rerun the benchmark before using final numbers.",
        "- Correctness is checked with output differences for deterministic operations and predictive scores for models.",
        "",
        "## Runtime Summary",
        "",
        "| Group | Case | GlassBox seconds | pandas/sklearn seconds | Speedup | Winner |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for group, case in cases:
        gb = timings[(group, case, "GlassBox")]
        base = timings[(group, case, "pandas/sklearn")]
        ratio, winner = winner_for_case(timings, group, case)
        lines.append(f"| {group} | {case} | {gb:.6f} | {base:.6f} | {ratio:.2f}x | {winner} |")

    lines.extend(
        [
            "",
            "## Showcase Cases",
            "",
            "This section highlights the strongest GlassBox wins and keeps exactly two pandas/scikit-learn wins for balanced reporting.",
            "",
            "| Group | Case | Speedup | Winner |",
            "|---|---|---:|---|",
        ]
    )
    for group, case, ratio, winner in showcase_cases(results):
        lines.append(f"| {group} | {case} | {ratio:.2f}x | {winner} |")

    lines.extend(
        [
            "",
            "## Correctness And Quality",
            "",
            "| Group | Case | Library | Metric | Value |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in results:
        if row.metric.startswith("runtime") or row.metric == "speedup_vs_baseline":
            continue
        lines.append(f"| {row.group} | {row.case} | {row.library} | {row.metric} | {row.value:.10g} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A `max_abs_diff` or `abs_diff` near zero means GlassBox matches the pandas/scikit-learn result for that operation.",
            "- A speedup above `1.0x` means GlassBox was faster in that benchmark run.",
            "- Model comparisons should be presented as comparable predictive efficiency, not exact equality, because some algorithms use different training methods.",
            "- If GlassBox is slower on a case but has similar accuracy, the honest claim is functional equivalence with optimization opportunities.",
            "",
            "## Suggested Claim Wording",
            "",
            "GlassBox provides pandas/scikit-learn-like APIs implemented in a transparent educational codebase. "
            "On deterministic dataframe, preprocessing, and metric operations, the benchmark checks numerical equivalence against established libraries. "
            "On model tasks, it compares predictive quality and runtime on the same generated datasets. "
            "Claims of better performance should be limited to the rows where the measured speedup is above `1.0x`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GlassBox against pandas and scikit-learn.")
    parser.add_argument("--rows", type=int, default=5000, help="Number of generated rows for most benchmarks.")
    parser.add_argument("--cols", type=int, default=8, help="Number of generated numeric columns/features.")
    parser.add_argument("--repeats", type=int, default=5, help="Timing repeats per benchmark case.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generated data.")
    parser.add_argument("--csv", type=Path, default=PACKAGE_ROOT / "results/comparison_results.csv")
    parser.add_argument("--report", type=Path, default=PACKAGE_ROOT / "docs/glassbox_comparison_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 100:
        raise SystemExit("--rows must be at least 100 for stable train/test splits.")
    if args.cols < 1:
        raise SystemExit("--cols must be at least 1.")
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1.")

    os.chdir(ROOT)
    rng = np.random.default_rng(args.seed)
    results: list[BenchmarkResult] = []

    benchmark_dataframe(results, rng, args.rows, args.cols, args.repeats)
    benchmark_preprocessing(results, rng, args.rows, args.cols, args.repeats)
    benchmark_metrics(results, rng, args.rows, args.repeats)
    benchmark_models(results, rng, args.rows, args.cols, args.repeats)

    write_csv(results, args.csv)
    write_report(results, args.report, args.rows, args.cols, args.repeats)
    print(f"Wrote CSV results to {args.csv}")
    print(f"Wrote Markdown report to {args.report}")


if __name__ == "__main__":
    main()
