import sys
import os
import json
import argparse
import traceback  # Added for deep error tracking
import numpy as np

sys.path.insert(0, os.path.abspath("packages/glassbox-numpandas/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-eda/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-preprocessing/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-ml/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-split/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-pipeline/src"))
sys.path.insert(0, os.path.abspath("packages/glassbox-optimization/src"))

from GlassBox.numpandas import read_csv
from GlassBox.eda import DataProfiler
from GlassBox.preprocessing import StandardScaler, SimpleImputer
from GlassBox.pipeline import Pipeline
from GlassBox.optimization import GridSearchCV
from GlassBox.ml.linear_model import LogisticRegressionGD


def run_automl(csv_path, target):
    try:
        # Load Data
        df = read_csv(csv_path)

        # I. Automated EDA (The Inspector)
        profiler = DataProfiler(df)
        stats = profiler.profile  # REMOVED THE () HERE

        # II. Preprocessing Engine (The Cleaner)
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegressionGD()),
            ]
        )

        # III. Parameter Optimization (The Orchestrator)
        X = df.drop(target)
        y = df[target]

        param_grid = {"clf__learning_rate": [0.01, 0.1, 0.5]}
        grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3)
        grid.fit(X, y)

        # IV. Explainability Extraction
        best_model = grid.best_estimator_.steps[-1][1]
        importance = {}
        if hasattr(best_model, "weights"):
            importance = dict(zip(X.columns, best_model.weights.tolist()))

        # V. JSON Report for Agent
        report = {
            "status": "success",
            "metrics": {"best_cv_score": grid.best_score_, "params": grid.best_params_},
            "explainability": {
                "feature_importance": importance,
                "top_feature": (
                    max(importance, key=importance.get) if importance else None
                ),
            },
            "data_profile": (
                {col: stats[col].get("type", "unknown") for col in stats}
                if isinstance(stats, dict)
                else "Profile not a dictionary"
            ),
        }
        return report

    except Exception as e:
        # Upgraded error handler to show exactly where it fails
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--target", required=True, help="Name of target column")
    args = parser.parse_args()

    results = run_automl(args.data, args.target)
    print(json.dumps(results, indent=4))
