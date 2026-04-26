# glassbox-benchmark

Benchmark and report utilities for comparing GlassBox with pandas and scikit-learn.

Run from the repository root without installing the package:

```powershell
python -m pip install -r requirements-dev.txt
python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py
```

Outputs:

- `packages/glassbox-benchmark/results/comparison_results.csv`
- `packages/glassbox-benchmark/docs/glassbox_comparison_report.md`

Use smaller or larger generated datasets with:

```powershell
python packages/glassbox-benchmark/src/GlassBox/benchmark/compare.py --rows 10000 --cols 12 --repeats 7
```

If the package is installed, you can also run:

```powershell
glassbox-compare --rows 10000 --cols 12 --repeats 7
```

The report is intentionally evidence-based: it shows numerical differences, predictive scores, runtime, and which library won each benchmark case.
