"""
Experiment 03: Hyperparameter search contribution.

Purpose:
Estimate how much performance gain comes from Random Forest hyperparameter tuning.

Compared models:
1. Baseline RF with fixed/simple parameters
2. Tuned RF using grouped cross-validation inside the training set

Protocol:
- Peak fluorescence threshold > 2000 a.u.
- Grouped raw-file train/validation/test split
- Uses the selected improved feature set:
    spectrometer + lifetime + size + time_asymmetry
- Selects tuned hyperparameters by grouped CV balanced accuracy
- Reports final performance on held-out test set

Primary question:
How much does hyperparameter tuning improve balanced accuracy?
"""

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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from lif_thesis.data.splits import make_group_split


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp03_hyperparameter_search")


def to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def peak_fluorescence(spectrum) -> float:
    arr = to_array(spectrum)
    return float(np.max(arr))


def stack_vector_column(df: pd.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].apply(to_array).values)


def build_feature_matrix(
    df: pd.DataFrame,
    scalar_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Uses:
    - spectrometer
    - lifetime
    - size
    - time_asymmetry
    """
    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    parts = []
    names = []

    X_spec = stack_vector_column(df, "spectrometer")
    parts.append(X_spec)
    names.extend([f"spectrometer_{i}" for i in range(X_spec.shape[1])])

    X_life = stack_vector_column(df, "lifetime")
    parts.append(X_life)
    names.extend([f"lifetime_{i}" for i in range(X_life.shape[1])])

    X_scalar = df[scalar_cols].copy()

    for col in scalar_cols:
        X_scalar[col] = pd.to_numeric(X_scalar[col], errors="coerce")

    if X_scalar.isna().any().any():
        raise ValueError(
            "Missing scalar values detected:\n"
            f"{X_scalar.isna().sum()}"
        )

    X_scalar = X_scalar.to_numpy(dtype=float)
    parts.append(X_scalar)
    names.extend(scalar_cols)

    X = np.concatenate(parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or Inf.")

    return X, names


def evaluate_model(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
) -> dict:
    y_pred = model.predict(X)

    labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_.astype(str)

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
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


def train_simple_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
) -> RandomForestClassifier:
    """
    Untuned baseline RF.
    """
    model = RandomForestClassifier(
        n_estimators=500,
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


def train_tuned_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    random_state: int,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict]:
    """
    Tuned RF using grouped CV.

    The grouped CV prevents particles from the same raw file from appearing
    in both the training and validation fold during tuning.
    """

    base_model = RandomForestClassifier(
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )

    param_grid = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 20, 40, 60],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"],
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
        return_train_score=True,
    )

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    cv_results = pd.DataFrame(grid_search.cv_results_)

    return (
        grid_search.best_estimator_,
        cv_results,
        {
            "best_params": grid_search.best_params_,
            "best_grouped_cv_balanced_accuracy": grid_search.best_score_,
        },
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "species"
    group_col = "raw_file"
    scalar_cols = ["size", "time_asymmetry"]
    fluorescence_threshold = 2000
    random_state = 42

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    required_cols = [
        label_col,
        group_col,
        "spectrometer",
        "lifetime",
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
    ].reset_index(drop=True)

    for col in scalar_cols:
        df = df[df[col].notna()].reset_index(drop=True)

    print(f"Particles before fluorescence threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print("\nClass counts:")
    print(df[label_col].value_counts())

    print("\nRaw files per class:")
    print(df.groupby(label_col)[group_col].nunique())

    X, feature_names = build_feature_matrix(
        df,
        scalar_cols=scalar_cols,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col=label_col,
        group_col=group_col,
        train_size=0.60,
        val_size=0.20,
        test_size=0.20,
        stratify=True,
        random_state=random_state,
        verbose=True,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    groups_train = df.iloc[train_idx][group_col].values

    print("\nTraining simple baseline RF...")
    simple_model = train_simple_rf(
        X_train,
        y_train,
        random_state=random_state,
    )

    print("\nTraining tuned RF with grouped CV...")
    tuned_model, cv_results, tuning_summary = train_tuned_rf(
        X_train,
        y_train,
        groups_train=groups_train,
        random_state=random_state,
    )

    cv_results.to_csv(OUTPUT_DIR / "grid_search_cv_results.csv", index=False)

    simple_metrics = {
        "train": evaluate_model(
            simple_model, X_train, y_train, label_encoder, "train"
        ),
        "val": evaluate_model(
            simple_model, X_val, y_val, label_encoder, "val"
        ),
        "test": evaluate_model(
            simple_model, X_test, y_test, label_encoder, "test"
        ),
    }

    tuned_metrics = {
        "train": evaluate_model(
            tuned_model, X_train, y_train, label_encoder, "train"
        ),
        "val": evaluate_model(
            tuned_model, X_val, y_val, label_encoder, "val"
        ),
        "test": evaluate_model(
            tuned_model, X_test, y_test, label_encoder, "test"
        ),
    }

    comparison = {
        "experiment": {
            "name": "exp03_hyperparameter_search",
            "label_col": label_col,
            "group_col": group_col,
            "features": ["spectrometer", "lifetime", *scalar_cols],
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "tuning_protocol": "StratifiedGroupKFold on training set",
            "primary_question": "How much gain comes from hyperparameter tuning?",
        },
        "tuning_summary": tuning_summary,
        "simple_rf": simple_metrics,
        "tuned_rf": tuned_metrics,
        "test_improvement": {
            "accuracy_delta": tuned_metrics["test"]["accuracy"]
            - simple_metrics["test"]["accuracy"],
            "balanced_accuracy_delta": tuned_metrics["test"]["balanced_accuracy"]
            - simple_metrics["test"]["balanced_accuracy"],
            "macro_f1_delta": tuned_metrics["test"]["macro_f1"]
            - simple_metrics["test"]["macro_f1"],
        },
    }

    with open(OUTPUT_DIR / "metrics_comparison.json", "w") as f:
        json.dump(comparison, f, indent=4)

    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    joblib.dump(simple_model, OUTPUT_DIR / "simple_rf.joblib")
    joblib.dump(tuned_model, OUTPUT_DIR / "tuned_rf.joblib")
    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.joblib")

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")

    print("\nSimple RF test performance:")
    print(
        json.dumps(
            {
                "accuracy": simple_metrics["test"]["accuracy"],
                "balanced_accuracy": simple_metrics["test"]["balanced_accuracy"],
                "macro_f1": simple_metrics["test"]["macro_f1"],
            },
            indent=4,
        )
    )

    print("\nTuned RF test performance:")
    print(
        json.dumps(
            {
                "accuracy": tuned_metrics["test"]["accuracy"],
                "balanced_accuracy": tuned_metrics["test"]["balanced_accuracy"],
                "macro_f1": tuned_metrics["test"]["macro_f1"],
            },
            indent=4,
        )
    )

    print("\nTuning improvement:")
    print(json.dumps(comparison["test_improvement"], indent=4))


if __name__ == "__main__":
    main()