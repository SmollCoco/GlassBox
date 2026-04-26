# GlassBox Pipeline

Pipeline sequential execution utilities orchestrating modular transformer coupling seamlessly mapped ending via a classifier or regressor instance safely wrapping underlying variables without leakage boundaries.

## API Usage

```python
from GlassBox.pipeline.pipeline import Pipeline
from GlassBox.preprocessing.impute import SimpleImputer
from GlassBox.ml.random_forest import RandomForest

# Stack components sequentially bypassing manual extraction steps!
pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', RandomForest(n_estimators=10))
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.
