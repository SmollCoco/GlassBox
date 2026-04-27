# GlassBox Ecosystem

This is the meta umbrella package. By installing `glassbox-meta`, you are triggering the resolution matrix fetching every underlying component representing the full ML Pipeline spectrum natively.

## Installation Usage
Just deploy the meta package dependencies across your python matrix.
```bash
pip install glassbox-meta
```


## Building Context
If you prefer developing internally, simply navigate inside your root monorepo target and use valid standard tools:
```bash
pip install -e packages/glassbox-meta
```

## Included Packages
- glassbox-autofit
- glassbox-benchmark
- glassbox-numpandas
- glassbox-eda
- glassbox-preprocessing
- glassbox-ml
- glassbox-eval
- glassbox-optimization
- glassbox-pipeline
- glassbox-split