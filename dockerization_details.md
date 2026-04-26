# GlassBox AutoFit: Containerized AutoML Engine

## 💡 Overview

GlassBox AutoFit is a zero-dependency, pure-NumPy Automated Machine Learning (AutoML) pipeline. It acts as a "White-Box" AI engine designed to be run inside an isolated Docker sandbox.

Instead of relying on black-box external libraries like Scikit-Learn, this engine builds, trains, and tunes models entirely from scratch using custom math. You provide a dataset and a target column; AutoFit handles the Exploratory Data Analysis (EDA), feature preprocessing, model selection, hyperparameter tuning, and returns a fully parsed, LLM-ready JSON report containing the metrics and feature importances.

This package serves as the core execution "Skill" for the NemoClaw Agent framework.

## ⚙️ How It Works (The Pipeline)

When triggered via the CLI (`GlassBox.autofit.cli`), the engine executes a deterministic, four-stage pipeline:

- **The Inspector (`DataProfiler` & `detect_task`)**: Automatically analyzes the incoming CSV. It detects whether the task is classification or regression based on the target column's variance and data types. It calculates missing values, detects outliers via IQR, and generates correlation matrices.
- **The Cleaner (`Pipeline` & `Preprocessing`)**: Dynamically constructs a transformation pipeline. It routes numeric data through a `SimpleImputer` (mean) and `StandardScaler`, while categorical data is routed through a `most_frequent` imputer and `OneHotEncoder`.
- **The Orchestrator (`RandomizedSearchCV` & Models)**: Iterates through an algorithmic registry (Logistic Regression, KNN, Decision Trees, Random Forests, Naive Bayes). It trains each scratch-built model, tunes them against a predefined hyperparameter grid using 5-fold cross-validation, and ranks them.
- **The Reporter (`report.py`)**: Consolidates the EDA statistics, preprocessing steps, and cross-validation scores for every model into a strictly structured JSON payload.

## 🐳 Architecture & the Dockerfile Explained

Due to the complex, nested `src/` layout of the GlassBox monorepo, standard pip installation (via setuptools auto-discovery) fails to map the namespaces correctly.

To solve this and ensure a lightning-fast build with zero `ModuleNotFoundError` crashes, the container relies on absolute path injection.

Here is exactly what the Dockerfile does to guarantee stability:

- `FROM python:3.11-slim`: Uses a lightweight base image to keep the container footprint small and build times fast.
- `RUN useradd -m -s /bin/bash agentuser`: Security hardening. Creates a non-root user so the NemoClaw agent executes code in a restricted, unprivileged state.
- `WORKDIR /app` and `RUN mkdir /data /results`: Establishes the core directories and grants the non-root user ownership of the output folders.
- `COPY . /app/`: Copies the entire GlassBox monorepo into the container. *(Note: The `.dockerignore` file prevents dynamic data like `test_data/` from bloating the build).*
- `ENV PYTHONPATH="..."`: The core architectural workaround. Instead of fighting pip auto-discovery, we explicitly map every single `src` folder across all 10 GlassBox packages directly into Python's internal path list. This stitches the monorepo together in memory instantly.
- `USER agentuser`: Drops root privileges before executing any code.

## 🚀 Usage

Do not run these modules directly on your local host environment. Always execute them through the Docker container to ensure environment consistency and security.

### 1. Build the Engine

Run this command from the root of the repository to compile the Docker image: