# glassbox-benchmark

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) ![NumPy-only Core](https://img.shields.io/badge/core-NumPy--only-2f9e44) ![Part of GlassBox](https://img.shields.io/badge/ecosystem-GlassBox-0b7285)

`glassbox-benchmark` provides reproducible benchmark utilities that compare selected GlassBox operations against pandas/scikit-learn baselines, then emits CSV results and a Markdown report for transparent performance and correctness checks.

## Installation

```bash
pip install glassbox-benchmark
```

## Import Example

```python
from GlassBox.benchmark import compare
```

## Minimal Usage

```python
import numpy as np
from pathlib import Path
from GlassBox.benchmark.compare import (
    BenchmarkResult,
    benchmark_dataframe,
    benchmark_metrics,
    write_report,
)

rng = np.random.default_rng(42)
results: list[BenchmarkResult] = []

benchmark_dataframe(results, rng, rows=1000, cols=6, repeats=3)
benchmark_metrics(results, rng, rows=1000, repeats=3)
write_report(results, path=Path("benchmark_report.md"), rows=1000, cols=6, repeats=3)
```

## API Inventory

| Class | Purpose |
|---|---|
| `BenchmarkResult` | Dataclass row model for benchmark outputs (group, case, metric, runtime, notes). |
| `timed(callable_: Callable[[], object], repeats: int) -> tuple[object, float, float]` | Measure repeated runtime and return result, mean time, and stddev. |
| `speedup(glassbox_seconds: float, baseline_seconds: float) -> float` | Compute GlassBox speedup ratio versus baseline runtime. |
| `add_timing_pair(results: list[BenchmarkResult], group: str, case: str, glassbox_seconds: float, baseline_seconds: float, glassbox_std: float, baseline_std: float) -> None` | Append comparable timing rows for GlassBox and baseline. |
| `benchmark_dataframe(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None` | Benchmark DataFrame reductions and numerical agreement. |
| `benchmark_preprocessing(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None` | Benchmark preprocessing scaling against sklearn. |
| `make_regression_data(rng: np.random.Generator, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]` | Generate synthetic regression dataset. |
| `make_classification_data(rng: np.random.Generator, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]` | Generate synthetic binary classification dataset. |
| `benchmark_metrics(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, repeats: int) -> None` | Benchmark metrics runtime and value differences. |
| `benchmark_models(results: list[BenchmarkResult], rng: np.random.Generator, rows: int, cols: int, repeats: int) -> None` | Benchmark model training/prediction tasks. |
| `write_csv(results: list[BenchmarkResult], path: Path) -> None` | Persist benchmark rows to CSV. |
| `runtime_lookup(results: list[BenchmarkResult]) -> dict[tuple[str, str, str], float]` | Build `(group, case, library) -> runtime` lookup table. |
| `winner_for_case(timings: dict[tuple[str, str, str], float], group: str, case: str) -> tuple[float, str]` | Determine winner and speedup for one benchmark case. |
| `showcase_cases(results: list[BenchmarkResult], baseline_limit: int = 2) -> list[tuple[str, str, float, str]]` | Select report highlight cases. |
| `write_report(results: list[BenchmarkResult], path: Path, rows: int, cols: int, repeats: int) -> None` | Write Markdown benchmark summary report. |
| `parse_args() -> argparse.Namespace` | Parse CLI arguments. |
| `main() -> None` | CLI entrypoint for full benchmark run. |

## Repository

Main GlassBox GitHub repository: [https://github.com/SmollCoco/GlassBox](https://github.com/SmollCoco/GlassBox)
