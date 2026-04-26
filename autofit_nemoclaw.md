# GlassBox AutoFit: Comprehensive Technical Feature Report

This report serves as a detailed technical overview of the AutoFit feature, a zero-dependency, pure-NumPy AutoML engine integrated into the GlassBox ecosystem and orchestrated via the NemoClaw AI agent framework.

## 1. Feature Overview and Objectives

The AutoFit feature provides an automated, end-to-end machine learning workflow within a secure, isolated sandbox. It is designed to take raw CSV data and a target column as input and automatically perform Exploratory Data Analysis (EDA), data cleaning, model selection, and hyperparameter tuning to produce a high-performing model and a diagnostic report.

## 2. Technical Architecture: File-by-File Breakdown

The logic is housed in the GlassBox.autofit package. Below is a description of the core files created and their roles.

- __init__.py:
  Acts as the package entry point, exposing the primary autofit function to the rest of the GlassBox ecosystem and external tools.

- core.py (The Central Engine):
  - Contains the autofit() master function which orchestrates the entire workflow.
  - Manages data splitting into training and testing sets.
  - Implements the feature preprocessing logic and assembles a final `Pipeline` with `SimpleImputer`, `StandardScaler`, and the selected estimator.
  - Executes the model training loop, triggering RandomizedSearchCV for hyperparameter optimization.
  - Returns a tuple `(report, fitted_pipeline)`.
  - Supports model serialization via `output_path` and writes a pickle artifact (default: `/results/best_model.pkl`).

- detect.py (Task Identification):
  - Houses the detect_task() function, which inspects the target column's data type and unique value counts.
  - Automatically labels the problem as "classification" (for example, boolean or few unique integers) or "regression" (for example, floating-point or many unique values).

- registry.py (Model and Parameter Management):
  - Maintains a mapping of model names to their scratch-built classes, such as LogisticRegressionGD, RandomForest, and GaussianNaiveBayes.
  - Defines the _DEFAULT_SEARCH_SPACE for each algorithm, specifying the ranges for hyperparameters like learning_rate and max_depth.

- report.py (Output Formatting):
  - Defines the build_report() function to ensure the final AutoML output follows a deterministic, JSON-serializable structure.
  - Identifies the best_model based on cross-validation scores.

- cli.py (Command-Line Entrypoint):
  - Exposes `--data`, `--target`, and `--output` flags.
  - Prints the JSON report while saving the fitted model pipeline to the output path.

## 3. Implementation Workflow and Commands

### Phase 1: Containerization and Pathing

To ensure stability across the complex monorepo structure, we implemented a custom Docker environment.

- Build command:
  ```bash
  docker build -t glassbox-env:latest .
  ```

- Logic:
  We used absolute PYTHONPATH injection in the Dockerfile to stitch together all internal packages (for example, glassbox-ml and glassbox-numpandas) without requiring fragile local installations.

### Phase 2: NemoClaw Infrastructure (WSL)

NemoClaw requires a Linux-based environment for its secure OpenShell sandbox.

- Installation:
  ```bash
  curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
  ```

- Onboarding:
  ```bash
  nemoclaw onboard
  ```

- Choice:
  Selected Google Gemini 2.5 Flash as the inference engine for its strong reasoning and tool-calling capabilities.

### Phase 3: Agent Integration

We bridged the AI agent to the Dockerized engine.

- Skill schema:
  Defined the GLASSBOX_TOOL_SCHEMA (parameters: csv_filename and target_column) in glassbox_tool.py.

- Library hardening:
  Manually patched openclaw/__init__.py to remove a broken TimeoutError import, resolving a known release bug in the OpenClaw SDK.

## 4. Design Choices and Rationale

- Zero-Dependency (NumPy Only):
  We avoided libraries like Scikit-Learn to ensure the pipeline remains a "White-Box" system where every mathematical operation is transparent and implemented from scratch.

- Absolute Path Injection:
  By setting PYTHONPATH in the Dockerfile, we bypassed pip auto-discovery issues common in nested src/ monorepos.

- Isolated Sandbox:
  All executions occur in a "deny-by-default" sandbox, preventing the agent from making unauthorized network requests or accessing sensitive host files.

## 5. Final Result and Output Summary

The engine produces a structured JSON report including:

- EDA:
  Shape, data types, and IQR-based outlier detection.

- Preprocessing:
  Summary of imputers and scalers applied.

- Metrics:
  Accuracy, Precision, Recall, and F1 for classification; MAE, MSE, and R2 for regression.

- Tuning:
  The best_params discovered for every tested algorithm.

- Serialized Model:
  A fitted pipeline artifact written to `/results/best_model.pkl` (or custom `--output` path).

## 6. Execution Guide

### Developer Manual Run

```powershell
docker run --rm `
  -v "${PWD}\test_data:/data" `
  -v "${PWD}\results:/results" `
  glassbox-env:latest `
  python -m GlassBox.autofit.cli --data /data/your_file.csv --target your_column --output /results/best_model.pkl
```

From the `agent/` folder, mount parent folders:

```powershell
docker run --rm `
  -v "${PWD}\..\test_data:/data" `
  -v "${PWD}\..\results:/results" `
  glassbox-env:latest `
  python -m GlassBox.autofit.cli --data /data/your_file.csv --target your_column --output /results/best_model.pkl
```

### Agent Interaction

Run python run_agent.py inside agent/ directory and prompt the assistant:
```bash
cd agent
python run_agent.py
```

"Analyze the 'test_model.csv' and predict the 'target' column."

The agent tool mounts both `test_data` and `results`, and persists the serialized model to `results/best_model.pkl` on the host.