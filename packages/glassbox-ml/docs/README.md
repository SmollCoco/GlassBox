# GlassBox Machine Learning

Transparently implemented traditional machine learning algorithms seamlessly operating against `numpandas` structures or `numpy` mappings directly.

## Key Features
- **Trees**: `RandomForest`, `DecisionTree`
- **Probabilistic**: `GaussianNaiveBayes`
- **Distance-bound**: `KNNClassifier`, `KNNRegressor`
- **Algebraic**: Linear & Logistic Regressions

## API Usage

```python
from GlassBox.ml.random_forest import RandomForest
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

X = DataFrame({"feature_1": [1.5, 2.3, 3.1], "feature_2": [0.5, 1.2, 0.9]})
y = Series([0, 1, 0])

# Initialize and train the internal model manually 
model = RandomForest(n_estimators=100, max_depth=5)
model.fit(X, y)

# Gather predictions
predictions = model.predict(X)
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.
