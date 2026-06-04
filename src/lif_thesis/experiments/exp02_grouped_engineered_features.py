"""
Experiment 02: Grouped split + engineered scattering summary features.

Purpose:
Test whether adding engineered scattering-derived scalar features improves
multiclass bacterial identification.

Compared with Experiment 01:
- Same fluorescence threshold: peak fluorescence > 2000 a.u.
- Same grouped raw-file split
- Same Random Forest tuning structure
- Same paper-style feature sets

New in Experiment 02:
- Adds optional scalar features:
    size
    time_asymmetry

Primary question:
How much do engineered scattering features improve balanced accuracy?
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib
from itertools import product

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


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp02_grouped_engineered_features")


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


def crop_pad_scattering(
    scattering,
    n_acquisitions: int = 60,
    n_angles: int = 24,
) -> np.ndarray:
    """
    Paper-style scattering preprocessing.

    - crop to 30 us
    - 60 acquisitions
    - 24 angular channels
    - zero-pad shorter signals
    - normalize to [0, 1]
    """
    target_len = n_acquisitions * n_angles

    arr = to_array(scattering).astype(float).flatten()

    if len(arr) >= target_len:
        arr = arr[:target_len]
    else:
        arr = np.pad(arr, (0, target_len - len(arr)), mode="constant")

    max_val = arr.max()

    if max_val > 0:
        arr = arr / max_val

    return arr


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    scalar_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix.

    Supports feature names:
    - spectra
    - lifetime
    - scattering
    - scalar

    Example feature sets:
    - spectra+lifetime
    - spectra+lifetime+scalar
    - spectra+lifetime+scattering+scalar
    """
    if scalar_cols is None:
        scalar_cols = ["size", "time_asymmetry"]

    parts = []
    names = []

    tokens = feature_set.split("+")

    if "spectra" in tokens:
        X_spec = stack_vector_column(df, "spectrometer")
        parts.append(X_spec)
        names.extend([f"spectrometer_{i}" for i in range(X_spec.shape[1])])

    if "lifetime" in tokens:
        X_life = stack_vector_column(df, "lifetime")
        parts.append(X_life)
        names.extend([f"lifetime_{i}" for i in range(X_life.shape[1])])

    if "scattering" in tokens:
        X_scat = np.stack(
            df["scattering_image"].apply(crop_pad_scattering).values
        )
        parts.append(X_scat)
        names.extend([f"scattering_{i}" for i in range(X_scat.shape[1])])

    if "scalar" in tokens:
        missing_scalars = [col for col in scalar_cols if col not in df.columns]
        if missing_scalars:
            raise ValueError(f"Missing scalar columns: {missing_scalars}")

        X_scalar = df[scalar_cols].copy()

        for col in scalar_cols:
            X_scalar[col] = pd.to_numeric(X_scalar[col], errors="coerce")

        if X_scalar.isna().any().any():
            missing_summary = X_scalar.isna().sum()
            raise ValueError(
                "Missing scalar values detected:\n"
                f"{missing_summary[missing_summary > 0]}"
            )

        X_scalar = X_scalar.to_numpy(dtype=float)

        parts.append(X_scalar)
        names.extend(scalar_cols)

    if not parts:
        raise ValueError(f"No valid features selected for {feature_set}")

    X = np.concatenate(parts, axis=1)

    if not np.isfinite(X).all():
        raise ValueError(f"Feature matrix for {feature_set} contains NaN/Inf.")

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
        "scattering_image",
        *scalar_cols,
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
        & df["lifetime"].notna()
        & df["scattering_image"].notna()
    ].reset_index(drop=True)

    for col in scalar_cols:
        df = df[df[col].notna()].reset_index(drop=True)

    print(f"Particles before fluorescence threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print("\nClass counts after filtering:")
    print(df[label_col].value_counts())

    print("\nRaw files per class after filtering:")
    print(df.groupby(label_col)[group_col].nunique())

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

    feature_sets = [
        # Paper-style features without engineered scalars
        "spectra",
        "lifetime",
        "scattering",
        "spectra+lifetime",
        "spectra+scattering",
        "lifetime+scattering",
        "spectra+lifetime+scattering",

        # Same feature sets with engineered scalars
        "spectra+scalar",
        "lifetime+scalar",
        "scattering+scalar",
        "spectra+lifetime+scalar",
        "spectra+scattering+scalar",
        "lifetime+scattering+scalar",
        "spectra+lifetime+scattering+scalar",
    ]

    n_estimators_grid = [100, 300, 500]
    max_depth_grid = [None, 20, 40]

    tuning_results = []
    best_score = -np.inf
    best_model = None
    best_config = None
    best_feature_names = None
    best_X = None

    print("\nStarting RF tuning with engineered feature comparisons...")

    for feature_set, n_estimators, max_depth in product(
        feature_sets,
        n_estimators_grid,
        max_depth_grid,
    ):
        print(
            f"\nTraining RF | features={feature_set} | "
            f"n_estimators={n_estimators} | max_depth={max_depth}"
        )

        X, feature_names = build_feature_matrix(
            df,
            feature_set=feature_set,
            scalar_cols=scalar_cols,
        )

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
        val_macro_f1 = f1_score(
            y_val,
            y_val_pred,
            average="macro",
            zero_division=0,
        )

        result = {
            "feature_set": feature_set,
            "uses_engineered_scalars": "scalar" in feature_set.split("+"),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "val_accuracy": val_accuracy,
            "val_balanced_accuracy": val_balanced_accuracy,
            "val_macro_f1": val_macro_f1,
            "n_features": X.shape[1],
        }

        tuning_results.append(result)

        print(
            f"Val accuracy={val_accuracy:.4f} | "
            f"Val balanced accuracy={val_balanced_accuracy:.4f} | "
            f"Val macro F1={val_macro_f1:.4f}"
        )

        selection_score = val_balanced_accuracy

        if selection_score > best_score:
            best_score = selection_score
            best_model = model
            best_config = result
            best_feature_names = feature_names
            best_X = X

    tuning_df = pd.DataFrame(tuning_results)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)

    # Summarize best result per feature set
    feature_summary = (
        tuning_df.sort_values("val_balanced_accuracy", ascending=False)
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("val_balanced_accuracy", ascending=False)
    )

    feature_summary.to_csv(OUTPUT_DIR / "feature_set_summary.csv", index=False)

    print("\nBest result per feature set:")
    print(feature_summary[[
        "feature_set",
        "uses_engineered_scalars",
        "val_balanced_accuracy",
        "val_macro_f1",
        "val_accuracy",
        "n_estimators",
        "max_depth",
    ]])

    print("\nBest overall configuration:")
    print(best_config)

    X_train = best_X[train_idx]
    X_val = best_X[val_idx]
    X_test = best_X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    metrics = {
        "experiment": {
            "name": "exp02_grouped_engineered_features",
            "label_col": label_col,
            "group_col": group_col,
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "scalar_features": scalar_cols,
            "note": (
                "Compares paper-style feature sets with and without engineered "
                "scattering summary features."
            ),
        },
        "best_config": best_config,
        "train": evaluate_model(
            best_model,
            X_train,
            y_train,
            label_encoder,
            "train",
        ),
        "val_tuning": evaluate_model(
            best_model,
            X_val,
            y_val,
            label_encoder,
            "val_tuning",
        ),
        "test_final": evaluate_model(
            best_model,
            X_test,
            y_test,
            label_encoder,
            "test_final",
        ),
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(best_feature_names, f, indent=4)

    joblib.dump(best_model, OUTPUT_DIR / "engineered_features_rf.joblib")
    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.joblib")

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_tuning_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_final_idx.npy", test_idx)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")

    print("\nFinal test performance:")
    print(
        json.dumps(
            {
                "accuracy": metrics["test_final"]["accuracy"],
                "balanced_accuracy": metrics["test_final"]["balanced_accuracy"],
                "macro_f1": metrics["test_final"]["macro_f1"],
                "best_config": best_config,
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()