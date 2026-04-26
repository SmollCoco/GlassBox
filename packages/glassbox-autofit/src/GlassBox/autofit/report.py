from __future__ import annotations

from typing import Any


def build_report(
    task: str,
    eda_summary: dict[str, Any],
    preprocessing_summary: dict[str, Any],
    model_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final AutoFit report with a deterministic JSON-friendly shape."""
    models_out: list[dict[str, Any]] = []
    best_model_name = ""
    best_model_score = float("-inf")

    for raw in model_results:
        entry: dict[str, Any] = {
            "name": raw.get("name", ""),
            "best_params": raw.get("best_params", {}),
            "metrics": raw.get("metrics", {}),
            "cv_score": float(raw.get("cv_score", 0.0)),
        }
        if "error" in raw:
            entry["error"] = str(raw["error"])

        score = entry["cv_score"]
        if "error" not in entry and score > best_model_score:
            best_model_score = score
            best_model_name = entry["name"]

        models_out.append(entry)

    return {
        "task": task,
        "eda": {
            "shape": eda_summary.get("shape", []),
            "dtypes": eda_summary.get("dtypes", {}),
            "missing_per_column": eda_summary.get("missing_per_column", {}),
            "outliers_detected": eda_summary.get("outliers_detected", {}),
            "correlations": eda_summary.get("correlations", {}),
        },
        "preprocessing": {
            "imputer": preprocessing_summary.get("imputer", ""),
            "scaler": preprocessing_summary.get("scaler", ""),
            "encoders_applied": preprocessing_summary.get("encoders_applied", []),
        },
        "models": models_out,
        "best_model": best_model_name,
    }
