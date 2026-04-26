# GlassBox Optimization

Hyperparameter optimization suites ensuring robust convergence and cross-validation integrity without compromising testing constraints recursively.

## API Usage

```python
from GlassBox.optimization.search import GridSearchCV
from GlassBox.ml.random_forest import RandomForest

model = RandomForest()
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}

# Search via K-Fold cross validation internally wrapping ML components!
finder = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
finder.fit(X_train, y_train)

# Extracted best elements
print("Best Params: ", finder.best_params_)
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.
