# GlassBox AutoFit: Containerized AutoML Engine

## 💡 Overview

GlassBox AutoFit is a zero-dependency, pure-NumPy Automated Machine Learning (AutoML) pipeline. It acts as a "White-Box" AI engine designed to be run inside an isolated Docker sandbox.

Instead of relying on black-box external libraries like Scikit-Learn, this engine builds, trains, and tunes models entirely from scratch using custom math. You provide a dataset and a target column; AutoFit handles the Exploratory Data Analysis (EDA), feature preprocessing, model selection, hyperparameter tuning, and returns a fully parsed, LLM-ready JSON report containing the metrics and feature importances.

This package serves as the core execution "Skill" for the NemoClaw Agent framework.

## ⚙️ How It Works (The Pipeline)

When triggered via the CLI (`GlassBox.autofit.cli`), the engine executes a deterministic, four-stage pipeline:

1. **The Inspector (`DataProfiler` & `detect_task`)**: Automatically analyzes the incoming CSV. It detects whether the task is classification or regression based on the target column's variance and data types. It calculates missing values, detects outliers via IQR, and generates correlation matrices.
2. **The Cleaner (`Pipeline` & Preprocessing)**: Dynamically constructs a transformation pipeline. It routes numeric data through a `SimpleImputer` (mean) and `StandardScaler`, while categorical data is routed through a `most_frequent` imputer and `OneHotEncoder`.
3. **The Orchestrator (`RandomizedSearchCV` & Models)**: Iterates through an algorithmic registry (Logistic Regression, KNN, Decision Trees, Random Forests, Naive Bayes). It trains each scratch-built model, tunes them against a predefined hyperparameter grid using 5-fold cross-validation, and ranks them.
4. **The Reporter (`report.py`)**: Consolidates the EDA statistics, preprocessing steps, and cross-validation scores for every model into a strictly structured JSON payload.

## 🐳 Architecture & The Dockerfile Explained

Due to the complex, nested `src/` layout of the GlassBox monorepo, standard pip installation (via setuptools auto-discovery) fails to map the namespaces correctly.

To solve this and ensure a lightning-fast build with zero `ModuleNotFoundError` crashes, the container relies on absolute path injection.

Here is exactly what the Dockerfile does to guarantee stability:

- `FROM python:3.11-slim`: Uses a lightweight base image to keep the container footprint small and build times fast.
- `RUN useradd -m -s /bin/bash agentuser`: Security hardening. Creates a non-root user so the NemoClaw agent executes code in a restricted, unprivileged state.
- `WORKDIR /app` & `RUN mkdir /data /results`: Establishes the core directories and grants the non-root user ownership of the output folders.
- `COPY . /app/`: Copies the entire GlassBox monorepo into the container. (Note: The `.dockerignore` file prevents dynamic data like `test_data/` from bloating the build).
- `ENV PYTHONPATH="..."`: The core architectural workaround. Instead of fighting pip auto-discovery, we explicitly map every single `src` folder across all 10 GlassBox packages directly into Python's internal path list. This stitches the monorepo together in memory instantly.
- `USER agentuser`: Drops root privileges before executing any code.

## 🚀 Usage

Do not run these modules directly on your local host environment. Always execute them through the Docker container to ensure environment consistency and security.

### 1. Build the Engine

Run this command from the root of the repository to compile the Docker image:

```bash
docker build -t glassbox-env:latest .
```

### 2. Execute an AutoFit Run

Ensure you have a CSV file located in a local `test_data` folder. The command below mounts data and results folders and triggers the pipeline.

```bash
docker run --rm \
  -v "${PWD}\test_data:/data" \
  -v "${PWD}\results:/results" \
  glassbox-env:latest \
  python -m GlassBox.autofit.cli --data /data/test_model.csv --target target --output /results/best_model.pkl
```

### 3. The Output

The engine outputs:

- A JSON report to stdout with task type, EDA statistics, preprocessing summary, and model scores.
- A serialized fitted pipeline artifact at `/results/best_model.pkl` (or your custom `--output` path).

#### Output Example

```powershell
(.venv) PS C:\Users\utilisateur\S6\AI\GlassBox\GlassBox> docker run --rm `
>>   -v "${PWD}\test_data:/data" `
>>   -v "${PWD}\results:/results" `
>>   glassbox-env:latest `
>>   python -m GlassBox.autofit.cli --data /data/test_model.csv --target target --output /results/best_model.pkl
{
  "task": "classification",
  "eda": {
    "shape": [
      100,
      3
    ],
    "dtypes": {
      "feature1": "float64",
      "feature2": "float64",
      "target": "int64"
    },
    "missing_per_column": {
      "feature1": 0,
      "feature2": 0,
      "target": 0
    },
    "outliers_detected": {
      "feature1": 1,
      "feature2": 1,
      "target": 0
    },
    "correlations": {
      "feature1_feature2": 0.032322051078504895,
      "feature1_target": 0.5564306877023568,
      "feature2_target": 0.6303796587129592
    }
  },
  "preprocessing": {
    "imputer": "mean",
    "scaler": "StandardScaler",
    "encoders_applied": []
  },
  "models": [
    {
      "name": "LogisticRegression",
      "best_params": {
        "learning_rate": 0.1,
        "n_iterations": 2000
      },
      "metrics": {
        "accuracy": 0.95,
        "precision": 1.0,
        "recall": 0.9,
        "f1": 0.9473684210526316
      },
      "cv_score": 1.0
    },
    {
      "name": "KNNClassifier",
      "best_params": {
        "n_neighbors": 3,
        "distance_metric": "manhattan"
      },
      "metrics": {
        "accuracy": 0.85,
        "precision": 1.0,
        "recall": 0.7,
        "f1": 0.8235294117647058
      },
      "cv_score": 0.9625
    },
    {
      "name": "DecisionTreeClassifier",
      "best_params": {
        "max_depth": 5,
        "min_samples_split": 2
      },
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.9090909090909091,
        "recall": 1.0,
        "f1": 0.9523809523809523
      },
      "cv_score": 0.925
    },
    {
      "name": "RandomForestClassifier",
      "best_params": {
        "n_estimators": 25,
        "max_depth": 5,
        "max_features": "log2"
      },
      "metrics": {
        "accuracy": 0.8,
        "precision": 0.875,
        "recall": 0.7,
        "f1": 0.7777777777777777
      },
      "cv_score": 0.8125
    },
    {
      "name": "NaiveBayes",
      "best_params": {
        "var_smoothing": 1e-07
      },
      "metrics": {
        "accuracy": 0.85,
        "precision": 1.0,
        "recall": 0.7,
        "f1": 0.8235294117647058
      },
      "cv_score": 0.9625
    }
  ],
  "best_model": "LogisticRegression"
}
```