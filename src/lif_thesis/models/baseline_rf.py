from __future__ import annotations

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split


def _to_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def build_spectra_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
) -> np.ndarray:
    """
    Convert parsed spectrometer vectors into a 2D ML matrix.

    Output shape:
        (n_particles, n_spectral_features)
    """
    if spectra_col not in df.columns:
        raise ValueError(f"{spectra_col} not found in dataframe.")

    valid_mask = df[spectra_col].notna()
    if not valid_mask.all():
        raise ValueError(
            f"{(~valid_mask).sum()} rows have missing {spectra_col} values."
        )

    X = np.stack(df[spectra_col].apply(_to_array).values)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

    return X


def train_baseline_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Baseline Random Forest model.

    This is intentionally simple and defensible as an initial baseline.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> dict:
    """
    Evaluate classifier and return metrics.
    """
    y_pred = model.predict(X)

    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_.astype(str)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro"),
        "weighted_f1": f1_score(y, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def run_baseline_rf_experiment(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "raw_file",
    spectra_col: str = "spectrometer",
    output_dir: str | Path = "results/baseline_rf",
    random_state: int = 42,
):
    """
    Full baseline RF experiment.

    Steps:
    1. Filter usable rows
    2. Build spectra feature matrix
    3. Encode labels
    4. Create grouped train/val/test split
    5. Train RF
    6. Evaluate
    7. Save model, label encoder, metrics, and split indices
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [label_col, group_col, spectra_col]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
    ].reset_index(drop=True)

    print(f"Using {len(df)} valid particle rows.")
    print(f"Number of labels: {df[label_col].nunique()}")
    print(f"Number of groups: {df[group_col].nunique()}")

    X = build_spectra_matrix(df, spectra_col=spectra_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        stratify=True,
        random_state=random_state,
        verbose=True,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = train_baseline_rf(
        X_train,
        y_train,
        random_state=random_state,
    )

    metrics = {
        "train": evaluate_classifier(
            model, X_train, y_train, label_encoder, "train"
        ),
        "val": evaluate_classifier(
            model, X_val, y_val, label_encoder, "val"
        ),
        "test": evaluate_classifier(
            model, X_test, y_test, label_encoder, "test"
        ),
    }

    joblib.dump(model, output_dir / "baseline_rf.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {output_dir}")

    return model, label_encoder, metrics, (train_idx, val_idx, test_idx)