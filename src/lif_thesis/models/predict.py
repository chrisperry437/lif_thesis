"""
Prediction utilities for trained Rapid-E bioaerosol classifiers.

This module provides reusable inference logic for:
    - offline notebooks
    - real-time mock pipeline
    - FastAPI prediction service
    - Streamlit dashboard backend

It supports scikit-learn/joblib models by default and includes a lightweight
PyTorch path for saved torch models.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from lif_thesis.data.preprocessing import (
    PreprocessingConfig,
    build_feature_matrix,
    preprocess_particles,
)

logger = logging.getLogger(__name__)


ModelType = Literal["sklearn", "torch"]
TaskType = Literal["binary", "species", "multiclass"]


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for model inference."""

    model_path: str | Path
    model_type: ModelType = "sklearn"
    task: TaskType = "species"

    label_mapping_path: str | Path | None = None
    feature_columns_path: str | Path | None = None

    confidence_threshold: float = 0.60
    unknown_label: str = "unknown"

    preprocessing: PreprocessingConfig = PreprocessingConfig()


@dataclass
class PredictionResult:
    """Container for model prediction outputs."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    is_unknown: bool = False


def load_label_mapping(path: str | Path | None) -> dict[int, str] | None:
    """
    Load class-index-to-label mapping from JSON.

    Expected format:
        {
            "0": "control",
            "1": "B. cereus"
        }
    """
    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return {int(k): str(v) for k, v in raw.items()}


def load_feature_columns(path: str | Path | None) -> list[str] | None:
    """
    Load expected model feature columns.

    Supported formats:
        - JSON list
        - plain text file with one feature per line
    """
    if path is None:
        return None

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            features = json.load(f)
        return [str(x) for x in features]

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_model(path: str | Path, model_type: ModelType = "sklearn") -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    path:
        Model artifact path.
    model_type:
        Either "sklearn" or "torch".

    Returns
    -------
    Any
        Loaded model object.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if model_type == "sklearn":
        return joblib.load(path)

    if model_type == "torch":
        import torch

        model = torch.load(path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
        return model

    raise ValueError(f"Unsupported model_type: {model_type}")


def align_feature_columns(
    X: pd.DataFrame,
    expected_columns: list[str] | None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Align feature matrix to model training columns.

    Missing columns are filled with `fill_value`.
    Extra columns are dropped.
    """
    if expected_columns is None:
        return X

    X = X.copy()

    for col in expected_columns:
        if col not in X.columns:
            X[col] = fill_value

    return X[expected_columns]


def infer_class_labels(
    model: Any,
    label_mapping: dict[int, str] | None = None,
) -> list[str]:
    """
    Infer class labels from a model or optional label mapping.
    """
    if label_mapping is not None:
        return [label_mapping[i] for i in sorted(label_mapping)]

    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]

    raise ValueError(
        "Unable to infer class labels. Provide label_mapping_path in PredictionConfig."
    )


def predict_proba_sklearn(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return class probabilities from a scikit-learn style model.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)

        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])

        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    preds = np.asarray(model.predict(X))
    labels = list(model.classes_) if hasattr(model, "classes_") else sorted(set(preds))

    proba = np.zeros((len(preds), len(labels)), dtype=float)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    for i, pred in enumerate(preds):
        proba[i, label_to_idx[pred]] = 1.0

    return proba


def predict_proba_torch(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Return class probabilities from a PyTorch model.

    This assumes the torch model accepts a 2D tensor:
        batch_size x n_features

    For custom multimodal networks, create a wrapper model that accepts this
    flattened input or implement a model-specific prediction function.
    """
    import torch

    values = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X
    tensor = torch.as_tensor(values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)

        if isinstance(logits, tuple):
            logits = logits[0]

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    return probabilities


def probabilities_to_results(
    probabilities: np.ndarray,
    class_labels: list[str],
    confidence_threshold: float = 0.60,
    unknown_label: str = "unknown",
) -> list[PredictionResult]:
    """
    Convert probability array into structured prediction results.
    """
    results: list[PredictionResult] = []

    for row in probabilities:
        best_idx = int(np.argmax(row))
        confidence = float(row[best_idx])
        label = class_labels[best_idx]

        is_unknown = confidence < confidence_threshold

        result_label = unknown_label if is_unknown else label

        proba_dict = {
            class_labels[i]: float(row[i])
            for i in range(len(class_labels))
        }

        results.append(
            PredictionResult(
                label=result_label,
                confidence=confidence,
                probabilities=proba_dict,
                is_unknown=is_unknown,
            )
        )

    return results


class ParticlePredictor:
    """
    End-to-end particle predictor.

    This class handles:
        - model loading
        - preprocessing
        - feature alignment
        - probability prediction
        - confidence thresholding
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model = load_model(config.model_path, config.model_type)
        self.label_mapping = load_label_mapping(config.label_mapping_path)
        self.feature_columns = load_feature_columns(config.feature_columns_path)
        self.class_labels = infer_class_labels(self.model, self.label_mapping)

        logger.info("Loaded model from %s", config.model_path)
        logger.info("Class labels: %s", self.class_labels)

    def preprocess(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw particle dataframe.
        """
        processed = preprocess_particles(
            particles,
            config=self.config.preprocessing,
            keep_rejected=False,
        )

        X = build_feature_matrix(processed)
        X = align_feature_columns(X, self.feature_columns)

        return X

    def predict_features(self, X: pd.DataFrame) -> list[PredictionResult]:
        """
        Predict from an already-prepared feature matrix.
        """
        if self.config.model_type == "sklearn":
            probabilities = predict_proba_sklearn(self.model, X)
        elif self.config.model_type == "torch":
            probabilities = predict_proba_torch(self.model, X)
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}")

        return probabilities_to_results(
            probabilities=probabilities,
            class_labels=self.class_labels,
            confidence_threshold=self.config.confidence_threshold,
            unknown_label=self.config.unknown_label,
        )

    def predict_particles(self, particles: pd.DataFrame) -> pd.DataFrame:
        """
        Predict labels for a raw particle dataframe.

        Returns a dataframe containing the original accepted particles plus:
            - predicted_label
            - prediction_confidence
            - is_unknown
            - class probabilities
        """
        processed = preprocess_particles(
            particles,
            config=self.config.preprocessing,
            keep_rejected=False,
        )

        X = build_feature_matrix(processed)
        X = align_feature_columns(X, self.feature_columns)

        results = self.predict_features(X)

        output = processed.copy()

        output["predicted_label"] = [r.label for r in results]
        output["prediction_confidence"] = [r.confidence for r in results]
        output["is_unknown"] = [r.is_unknown for r in results]

        for class_label in self.class_labels:
            safe_col = (
                "proba_"
                + class_label.replace(" ", "_")
                .replace(".", "")
                .replace("/", "_")
                .replace("-", "_")
            )
            output[safe_col] = [
                r.probabilities.get(class_label, 0.0) for r in results
            ]

        return output


def predict_dataframe(
    particles: pd.DataFrame,
    model_path: str | Path,
    model_type: ModelType = "sklearn",
    label_mapping_path: str | Path | None = None,
    feature_columns_path: str | Path | None = None,
    confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Convenience function for predicting directly from a dataframe.
    """
    predictor = ParticlePredictor(
        PredictionConfig(
            model_path=model_path,
            model_type=model_type,
            label_mapping_path=label_mapping_path,
            feature_columns_path=feature_columns_path,
            confidence_threshold=confidence_threshold,
        )
    )

    return predictor.predict_particles(particles)


def summarize_predictions(
    predictions: pd.DataFrame,
    label_col: str = "predicted_label",
    confidence_col: str = "prediction_confidence",
) -> pd.DataFrame:
    """
    Summarize particle-level predictions into label counts and proportions.
    """
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "predicted_label",
                "count",
                "proportion",
                "mean_confidence",
            ]
        )

    total = len(predictions)

    summary = (
        predictions.groupby(label_col, dropna=False)
        .agg(
            count=(label_col, "size"),
            mean_confidence=(confidence_col, "mean"),
        )
        .reset_index()
        .rename(columns={label_col: "predicted_label"})
    )

    summary["proportion"] = summary["count"] / total

    return summary.sort_values("count", ascending=False).reset_index(drop=True)


__all__ = [
    "PredictionConfig",
    "PredictionResult",
    "ParticlePredictor",
    "load_model",
    "predict_dataframe",
    "summarize_predictions",
]