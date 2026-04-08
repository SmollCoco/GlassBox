# GlassBox Packaging Architecture

## Goals
- Use a true multi-package monorepo under a shared namespace.
- Keep package boundaries explicit and dependency direction clean.
- Build and release each package independently.

## Namespace Model
Installable packages expose a shared namespace:
- `GlassBox.numpandas`
- `GlassBox.eda`
- `GlassBox.preprocessing`
- `GlassBox.ml`

A meta package `glassbox` depends on all package distributions.

## Repository Layout

```text
packages/
  glassbox-numpandas/
    pyproject.toml
    src/GlassBox/numpandas/
    tests/
  glassbox-eda/
    pyproject.toml
    src/GlassBox/eda/
    tests/
  glassbox-preprocessing/
    pyproject.toml
    src/GlassBox/preprocessing/
    tests/
  glassbox-ml/
    pyproject.toml
    src/GlassBox/ml/
    tests/
  glassbox-meta/
    pyproject.toml
```

## Package Boundaries
- `glassbox-numpandas`
: Core tabular abstractions and IO.
- `glassbox-eda`
: Profiling, plotting, and descriptive statistics.
- `glassbox-preprocessing`
: Imputation, scaling, encoding, and composition.
- `glassbox-ml`
: Learning algorithms, metrics, and ML utilities.

## Dependency Graph

```text
glassbox-numpandas -> numpy, openpyxl
glassbox-eda -> glassbox-numpandas, numpy, matplotlib, seaborn
glassbox-preprocessing -> glassbox-numpandas, numpy
glassbox-ml -> numpy
glassbox (meta) -> all of the above
```

## Build Model
Build from each package directory:

```bash
python -m pip install --upgrade build
python -m build packages/glassbox-numpandas
python -m build packages/glassbox-eda
python -m build packages/glassbox-preprocessing
python -m build packages/glassbox-ml
python -m build packages/glassbox-meta
```

## Test Strategy
- Keep tests package-local under each package `tests/` directory.
- Run package tests against installed package or package `src` path.
- Keep integration tests separate from unit tests.

## Dependency Management
- Runtime dependencies live in each package `pyproject.toml` under `[project.dependencies]`.
- Shared runtime umbrella is listed in root `requirements.txt`.
- Development tools are isolated in root `requirements-dev.txt`.
