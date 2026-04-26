import subprocess
from pathlib import Path

# 1. The Schema: This tells the LLM how to use your tool
GLASSBOX_TOOL_SCHEMA = {
    "name": "glassbox_autofit",
    "description": "An AutoML engine that performs EDA, data cleaning, and trains 5 different models to predict a target column. Use this whenever the user asks to analyze a CSV or build a predictive model.",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_filename": {
                "type": "string",
                "description": "The exact name of the CSV file in the test_data folder (e.g., 'test_model.csv').",
            },
            "target_column": {
                "type": "string",
                "description": "The name of the column to predict (e.g., 'target').",
            },
        },
        "required": ["csv_filename", "target_column"],
    },
}


# 2. The Execution Function: This fires up your Docker container
def execute_glassbox(csv_filename: str, target_column: str) -> str:
    print(f"Agent triggered GlassBox for {csv_filename}...")

    # Resolve an absolute host path for reliable Docker volume mounting on Windows.
    workspace_root = Path(__file__).resolve().parents[1]
    test_data_dir = workspace_root / "test_data"
    results_dir = workspace_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    mount_source = str(test_data_dir).replace("\\", "/")
    results_mount_source = str(results_dir).replace("\\", "/")

    docker_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{mount_source}:/data",
        "-v",
        f"{results_mount_source}:/results",
        "-e",
        "PYTHONPATH=/app/packages/glassbox-autofit/src:/app/packages/glassbox-benchmark/src:/app/packages/glassbox-eda/src:/app/packages/glassbox-meta/src:/app/packages/glassbox-ml/src:/app/packages/glassbox-numpandas/src:/app/packages/glassbox-optimization/src:/app/packages/glassbox-pipeline/src:/app/packages/glassbox-preprocessing/src:/app/packages/glassbox-split/src",
        "glassbox-env:latest",
        "python",
        "-m",
        "GlassBox.autofit.cli",
        "--data",
        f"/data/{csv_filename}",
        "--target",
        target_column,
        "--output",
        "/results/best_model.pkl",
    ]

    result = subprocess.run(docker_command, capture_output=True, text=True)

    if result.returncode != 0:
        return f"Error running GlassBox: {result.stderr.strip()}"

    return result.stdout  # Returns the JSON report to the agent
