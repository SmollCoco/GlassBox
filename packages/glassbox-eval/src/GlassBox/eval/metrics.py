"""Classification metrics utilities for transparent model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=None)
    if arr.ndim == 0:
        raise ValueError(f"{name} must be a 1D array-like with at least one dimension.")
    return np.ravel(arr)


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except Exception:
        return False


def _label_equal(left: Any, right: Any) -> bool:
    if _is_nan(left) and _is_nan(right):
        return True
    return bool(left == right)


def _validate_targets(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true = _to_numpy_1d(y_true, name="y_true")
    pred = _to_numpy_1d(y_pred, name="y_pred")
    if true.shape[0] != pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return true, pred


def _ordered_unique(values: np.ndarray) -> list[Any]:
    unique_values: list[Any] = []
    for value in values.tolist():
        if not any(_label_equal(value, seen) for seen in unique_values):
            unique_values.append(value)
    return unique_values


def _resolve_labels(true: np.ndarray, pred: np.ndarray, labels: Any = None) -> list[Any]:
    if labels is None:
        return _ordered_unique(np.concatenate([true, pred]))

    resolved = _to_numpy_1d(labels, name="labels").tolist()
    if len(resolved) == 0:
        raise ValueError("labels must contain at least one class.")

    deduped: list[Any] = []
    for label in resolved:
        if not any(_label_equal(label, existing) for existing in deduped):
            deduped.append(label)
    return deduped


def _label_to_index(label: Any, labels: list[Any]) -> int:
    for idx, known in enumerate(labels):
        if _label_equal(label, known):
            return idx
    return -1


def confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: Any = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Compute the confusion matrix for classification predictions.

    Rows correspond to true labels, columns correspond to predicted labels.
    """
    true, pred = _validate_targets(y_true, y_pred)
    class_labels = _resolve_labels(true, pred, labels=labels)
    size = len(class_labels)
    matrix = np.zeros((size, size), dtype=float if normalize else int)

    for true_label, pred_label in zip(true, pred):
        i = _label_to_index(true_label, class_labels)
        j = _label_to_index(pred_label, class_labels)
        if i == -1 or j == -1:
            raise ValueError("Found label in y_true/y_pred that is not present in labels.")
        matrix[i, j] += 1

    if normalize is None:
        return matrix.astype(int)

    if normalize not in {"true", "pred", "all"}:
        raise ValueError("normalize must be one of None, 'true', 'pred', or 'all'.")

    matrix = matrix.astype(float)
    if normalize == "true":
        row_sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    if normalize == "pred":
        col_sums = matrix.sum(axis=0, keepdims=True)
        return np.divide(matrix, col_sums, out=np.zeros_like(matrix), where=col_sums != 0)

    total = matrix.sum()
    if total == 0:
        return np.zeros_like(matrix)
    return matrix / total


def binary_confusion_counts(
    y_true: Any,
    y_pred: Any,
    positive_label: Any = 1,
) -> dict[str, int]:
    """Return TP/TN/FP/FN counts for binary classification."""
    true, pred = _validate_targets(y_true, y_pred)

    tp = int(np.sum((true == positive_label) & (pred == positive_label)))
    fn = int(np.sum((true == positive_label) & (pred != positive_label)))
    fp = int(np.sum((true != positive_label) & (pred == positive_label)))
    tn = int(np.sum((true != positive_label) & (pred != positive_label)))

    return {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
    }


def _safe_divide(numerator: float, denominator: float, zero_division: float) -> float:
    if denominator == 0:
        return float(zero_division)
    return float(numerator / denominator)


def _format_classification_report(report: dict[str, Any], digits: int) -> str:
    headers = ("label", "precision", "recall", "f1-score", "support")
    label_width = max([len(headers[0]), *(len(name) for name in report.keys())])
    metric_width = max(9, digits + 4)

    lines = [
        f"{headers[0]:<{label_width}} "
        f"{headers[1]:>{metric_width}} "
        f"{headers[2]:>{metric_width}} "
        f"{headers[3]:>{metric_width}} "
        f"{headers[4]:>{metric_width}}"
    ]

    def add_row(name: str, values: dict[str, Any]) -> None:
        lines.append(
            f"{name:<{label_width}} "
            f"{values['precision']:{metric_width}.{digits}f} "
            f"{values['recall']:{metric_width}.{digits}f} "
            f"{values['f1-score']:{metric_width}.{digits}f} "
            f"{int(values['support']):>{metric_width}d}"
        )

    metric_rows = [
        key for key in report.keys() if key not in {"accuracy", "macro avg", "weighted avg"}
    ]
    for key in metric_rows:
        add_row(key, report[key])

    lines.append("")
    accuracy = report["accuracy"]
    support_total = int(report["weighted avg"]["support"])
    lines.append(
        f"{'accuracy':<{label_width}} "
        f"{'':>{metric_width}} {'':>{metric_width}} "
        f"{accuracy:{metric_width}.{digits}f} "
        f"{support_total:>{metric_width}d}"
    )
    add_row("macro avg", report["macro avg"])
    add_row("weighted avg", report["weighted avg"])

    return "\n".join(lines)


def classification_report(
    y_true: Any,
    y_pred: Any,
    labels: Any = None,
    digits: int = 4,
    zero_division: float = 0.0,
    output_dict: bool = True,
) -> dict[str, Any] | str:
    """Build a per-class precision/recall/F1 classification report.

    Returns a dict by default. Set output_dict=False to get a table string.
    """
    if digits < 0:
        raise ValueError("digits must be non-negative.")

    true, pred = _validate_targets(y_true, y_pred)
    class_labels = _resolve_labels(true, pred, labels=labels)

    report: dict[str, Any] = {}
    supports: list[int] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in class_labels:
        support = int(np.sum(true == label))
        tp = float(np.sum((true == label) & (pred == label)))
        fp = float(np.sum((true != label) & (pred == label)))
        fn = float(np.sum((true == label) & (pred != label)))

        precision = _safe_divide(tp, tp + fp, zero_division=zero_division)
        recall = _safe_divide(tp, tp + fn, zero_division=zero_division)
        f1 = _safe_divide(2 * precision * recall, precision + recall, zero_division=zero_division)

        key = str(label)
        report[key] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

        supports.append(support)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    total_support = int(np.sum(supports))
    accuracy = _safe_divide(float(np.sum(true == pred)), float(true.shape[0]), zero_division=zero_division)

    if len(class_labels) == 0:
        macro_precision = macro_recall = macro_f1 = float(zero_division)
    else:
        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

    if total_support == 0:
        weighted_precision = weighted_recall = weighted_f1 = float(zero_division)
    else:
        weighted_precision = float(np.average(precisions, weights=supports))
        weighted_recall = float(np.average(recalls, weights=supports))
        weighted_f1 = float(np.average(f1s, weights=supports))

    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1-score": macro_f1,
        "support": total_support,
    }
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": weighted_f1,
        "support": total_support,
    }

    if output_dict:
        return report
    return _format_classification_report(report, digits=digits)
