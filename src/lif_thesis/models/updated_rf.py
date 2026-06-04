## Baseline Random Forest classifier for LIF thesis experiments

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
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier


def _to_array(x) -> np.ndarray:
    """
    Convert list-like values stored in parquet into numpy arrays.
    """
    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, list):
        return np.asarray(x)

    return np.asarray(x)


def _safe_stack_vector_column(
    df: pd.DataFrame,
    col: str,
) -> np.ndarray:
    """
    Stack a dataframe column containing list-like vectors into a 2D matrix.
    """
    if col not in df.columns:
        raise ValueError(f"{col} not found in dataframe.")

    if df[col].isna().any():
        raise ValueError(f"{df[col].isna().sum()} rows have missing {col} values.")

    X = np.stack(df[col].apply(_to_array).values)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix for {col}, got shape {X.shape}")

    return X


def build_spectra_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
) -> np.ndarray:
    """
    Convert parsed spectrometer vectors into a 2D ML matrix.

    Output shape:
        (n_particles, n_spectral_features)
    """
    return _safe_stack_vector_column(df, spectra_col)


def build_multimodal_feature_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scalar_cols: list[str] | None = None,
    include_spectrometer: bool = True,
    include_lifetime: bool = True,
    include_scalars: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix from spectrometer, lifetime, and scalar features.

    Features included by default:
    - spectrometer vector
    - lifetime vector
    - size
    - time_asymmetry

    Returns
    -------
    X : np.ndarray
        Feature matrix.

    feature_names : list[str]
        Names corresponding to feature columns.
    """

    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    feature_parts = []
    feature_names = []

    if include_spectrometer:
        X_spec = _safe_stack_vector_column(df, spectra_col)
        feature_parts.append(X_spec)
        feature_names.extend(
            [f"{spectra_col}_{i}" for i in range(X_spec.shape[1])]
        )

    if include_lifetime:
        X_life = _safe_stack_vector_column(df, lifetime_col)
        feature_parts.append(X_life)
        feature_names.extend(
            [f"{lifetime_col}_{i}" for i in range(X_life.shape[1])]
        )

    if include_scalars:
        missing_scalars = [col for col in scalar_cols if col not in df.columns]
        if missing_scalars:
            raise ValueError(f"Missing scalar columns: {missing_scalars}")

        X_scalar = df[scalar_cols].copy()

        for col in scalar_cols:
            X_scalar[col] = pd.to_numeric(X_scalar[col], errors="coerce")

        if X_scalar.isna().any().any():
            missing_summary = X_scalar.isna().sum()
            raise ValueError(
                "Missing scalar values found:\n"
                f"{missing_summary[missing_summary > 0]}"
            )

        feature_parts.append(X_scalar.to_numpy(dtype=float))
        feature_names.extend(scalar_cols)

    if not feature_parts:
        raise ValueError("No feature groups selected.")

    X = np.concatenate(feature_parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    return X, feature_names


def train_baseline_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train Random Forest with grouped cross-validation.

    Groups prevent particles from the same raw_file appearing
    in both training and validation folds during hyperparameter tuning.
    """

    base_model = RandomForestClassifier(
        criterion="gini",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    param_grid = {
        "n_estimators": [300, 500],
        "max_depth": [None, 20, 40],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    print("\nBest RF parameters:")
    print(grid_search.best_params_)

    print("\nBest grouped-CV balanced accuracy:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_


def _safe_roc_auc(
    y: np.ndarray,
    y_proba: np.ndarray,
) -> float | None:
    """
    Compute ROC-AUC safely for binary or multiclass classification.
    """
    try:
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y, y_proba[:, 1]))

        return float(
            roc_auc_score(
                y,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        )
    except Exception:
        return None


def _safe_pr_auc(
    y: np.ndarray,
    y_proba: np.ndarray,
) -> float | None:
    """
    Compute PR-AUC safely for binary or multiclass classification.
    Uses macro average precision for multiclass.
    """
    try:
        n_classes = y_proba.shape[1]

        if n_classes == 2:
            return float(average_precision_score(y, y_proba[:, 1]))

        y_onehot = np.zeros_like(y_proba)

        for cls in range(n_classes):
            y_onehot[:, cls] = (y == cls).astype(int)

        return float(
            average_precision_score(
                y_onehot,
                y_proba,
                average="macro",
            )
        )
    except Exception:
        return None


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
    y_proba = model.predict_proba(X)

    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_.astype(str)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "roc_auc": _safe_roc_auc(y, y_proba),
        "pr_auc": _safe_pr_auc(y, y_proba),
        "confusion_matrix": confusion_matrix(
            y,
            y_pred,
            labels=labels,
        ).tolist(),
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
    label_col: str = "species",
    group_col: str = "raw_file",
    spectra_col: str = "spectrometer",
    lifetime_col: str = "lifetime",
    scalar_cols: list[str] | None = None,
    output_dir: str | Path = "results/baseline_rf_multimodal",
    random_state: int = 42,
):
    """
    Full baseline RF experiment.

    Features:
    - spectrometer
    - lifetime
    - size
    - time_asymmetry

    Steps:
    1. Filter usable rows
    2. Build multimodal feature matrix
    3. Encode labels
    4. Create grouped train/val/test split
    5. Train RF
    6. Evaluate
    7. Save model, label encoder, feature names, metrics, and split indices
    """

    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = [
        label_col,
        group_col,
        spectra_col,
        lifetime_col,
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    usable_mask = (
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
        & df[lifetime_col].notna()
    )

    for col in scalar_cols:
        usable_mask = usable_mask & df[col].notna()

    df = df[usable_mask].reset_index(drop=True)

    print(f"Using {len(df)} valid particle rows.")
    print(f"Number of labels: {df[label_col].nunique()}")
    print(f"Number of groups: {df[group_col].nunique()}")

    X, feature_names = build_multimodal_feature_matrix(
        df,
        spectra_col=spectra_col,
        lifetime_col=lifetime_col,
        scalar_cols=scalar_cols,
        include_spectrometer=True,
        include_lifetime=True,
        include_scalars=True,
    )

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

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

    groups_train = df.iloc[train_idx][group_col].values

    model = train_baseline_rf(
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
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

    joblib.dump(model, output_dir / "baseline_rf_multimodal.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {output_dir}")

    return model, label_encoder, metrics, (train_idx, val_idx, test_idx)