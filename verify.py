import sys
import os

# Add paths manually for test since we are not running via pytest
sys.path.insert(0, os.path.abspath('packages/glassbox-numpandas/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-eda/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-preprocessing/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-ml/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-split/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-pipeline/src'))
sys.path.insert(0, os.path.abspath('packages/glassbox-optimization/src'))

import numpy as np
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from GlassBox.split import train_test_split
from GlassBox.preprocessing import SMOTE, StandardScaler
from GlassBox.pipeline import Pipeline
from GlassBox.optimization import GridSearchCV
from GlassBox.ml.linear_model import LogisticRegressionGD

def test_pipeline():
    print("Testing pipeline end-to-end...")
    
    # Imbalanced Data
    X = DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'cat': np.random.choice([0, 1], 100)
    })
    
    y = Series(np.concatenate([np.zeros(90), np.ones(10)]), name='target')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('smote', SMOTE(k_neighbors=2, random_state=42)),
        ('clf', LogisticRegressionGD())
    ])
    
    param_grid = {
        'clf__learning_rate': [0.01, 0.1, 1.0]
    }
    
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2, scoring="accuracy")
    grid.fit(X_train, y_train)
    
    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)
    
    preds = grid.predict(X_test)
    print("Predictions len:", len(preds))
    print("Test finished successfully!")

if __name__ == "__main__":
    test_pipeline()
