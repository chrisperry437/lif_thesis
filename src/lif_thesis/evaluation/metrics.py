##Evaluation metrics (accuracy, precision, recall, F1, confusion matrices etc.)

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _safe_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
) -> float | None:
    """
    Compute ROC-AUC safely for binary or multiclass classification.
    Returns None if ROC-AUC cannot be computed.
    """
    try:
        classes = np.unique(y_true)

        if len(classes) == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))

        return float(
            roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average=average,
            )
        )

    except Exception:
        return None


def _safe_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    average: str = "macro",
) -> float | None:
    """
    Compute PR-AUC safely for binary or multiclass classification.
    Uses average precision score.
    """
    try:
        classes = np.unique(y_true)

        if len(classes) == 2:
            return float(average_precision_score(y_true, y_score[:, 1]))

        y_true_onehot = np.zeros_like(y_score)

        for i, cls in enumerate(classes):
            y_true_onehot[:, i] = (y_true == cls).astype(int)

        return float(
            average_precision_score(
                y_true_onehot,
                y_score,
                average=average,
            )
        )

    except Exception:
        return None


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    class_names: list[str] | None = None,
    average: str = "macro",
) -> dict[str, Any]:
    """
    Compute core classification metrics.

    Includes:
    - ROC-AUC
    - PR-AUC
    - Balanced accuracy
    - F1
    - Confusion matrix

    Parameters
    ----------
    y_true:
        True encoded labels.

    y_pred:
        Predicted encoded labels.

    y_proba:
        Predicted class probabilities from model.predict_proba(X).
        Required for ROC-AUC and PR-AUC.

    class_names:
        Optional class names in encoded order.

    average:
        Averaging method for multiclass F1, ROC-AUC, and PR-AUC.

    Returns
    -------
    dict
        Metrics dictionary.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [str(label) for label in labels]

    metrics = {
        "n_samples": int(len(y_true)),
        "n_classes": int(len(labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        f"{average}_f1": float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        metrics["roc_auc"] = _safe_roc_auc(
            y_true,
            y_proba,
            average=average,
        )

        metrics["pr_auc"] = _safe_pr_auc(
            y_true,
            y_proba,
            average=average,
        )

    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def metrics_to_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a metrics dictionary into a compact one-row DataFrame.
    Useful for comparing experiments.
    """

    row = {
        "n_samples": metrics.get("n_samples"),
        "n_classes": metrics.get("n_classes"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
    }

    return pd.DataFrame([row])


def compute_curves_binary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """
    Compute ROC and precision-recall curve data for binary classification.

    Assumes y_proba has shape:
        (n_samples, 2)

    Returns arrays converted to lists for JSON serialization.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2 or y_proba.shape[1] != 2:
        raise ValueError(
            "compute_curves_binary expects y_proba with shape (n_samples, 2)."
        )

    scores = y_proba[:, 1]

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)

    return {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "precision_recall_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def confusion_matrix_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return confusion matrix as a labeled DataFrame.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [str(label) for label in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    )