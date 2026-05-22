##Implements leakage-free data splits for training, validation and testing

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)


def validate_split_inputs(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
):
    """
    Validate split inputs.
    """
    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in dataframe.")

    if group_col not in df.columns:
        raise ValueError(f"{group_col} not found in dataframe.")

    if df[label_col].isna().any():
        raise ValueError(f"Missing values found in {label_col}.")

    if df[group_col].isna().any():
        raise ValueError(f"Missing values found in {group_col}.")


def summarize_split(
    df: pd.DataFrame,
    indices: np.ndarray,
    label_col: str,
    name: str,
):
    """
    Print useful split diagnostics.
    """
    subset = df.iloc[indices]

    print(f"\n{name}")
    print("-" * 60)
    print(f"N rows: {len(subset)}")
    print(f"N groups: {subset['__group__'].nunique()}")

    counts = subset[label_col].value_counts(normalize=True)

    print("\nClass distribution:")
    print(counts)


def make_group_split(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "raw_file",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Create grouped train/val/test splits.

    Features:
    ----------
    - Prevents leakage using group-based splitting
    - Attempts stratification when possible
    - Returns row indices for train/val/test

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    label_col : str
        Column used for stratification

    group_col : str
        Column defining leakage-safe groups
        (e.g. raw_file, run_id, experiment_id)

    train_size : float
        Fraction of data in training split

    val_size : float
        Fraction of data in validation split

    test_size : float
        Fraction of data in test split

    stratify : bool
        Attempt stratified grouped split

    random_state : int
        Random seed

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """

    validate_split_inputs(df, label_col, group_col)

    total = train_size + val_size + test_size

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train/val/test sizes must sum to 1. Got {total}"
        )

    df = df.copy()

    # Internal helper column
    df["__group__"] = df[group_col].astype(str)

    X = np.arange(len(df))
    y = df[label_col].values
    groups = df["__group__"].values

    # --------------------------------------------------------
    # STEP 1: Train vs temp split
    # --------------------------------------------------------

    if stratify:
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state,
            )

            splits = list(sgkf.split(X, y, groups))

            # Take first fold as temp
            train_idx, temp_idx = splits[0]

        except Exception as e:
            print(
                f"WARNING: StratifiedGroupKFold failed ({e}). "
                "Falling back to GroupShuffleSplit."
            )

            gss = GroupShuffleSplit(
                n_splits=1,
                train_size=train_size,
                random_state=random_state,
            )

            train_idx, temp_idx = next(
                gss.split(X, y, groups)
            )

    else:
        gss = GroupShuffleSplit(
            n_splits=1,
            train_size=train_size,
            random_state=random_state,
        )

        train_idx, temp_idx = next(
            gss.split(X, y, groups)
        )

    # --------------------------------------------------------
    # STEP 2: Split temp -> val/test
    # --------------------------------------------------------

    temp_df = df.iloc[temp_idx]

    temp_X = np.arange(len(temp_df))
    temp_y = temp_df[label_col].values
    temp_groups = temp_df["__group__"].values

    val_fraction_of_temp = val_size / (val_size + test_size)

    if stratify:
        try:
            sgkf_temp = StratifiedGroupKFold(
                n_splits=2,
                shuffle=True,
                random_state=random_state,
            )

            val_sub_idx, test_sub_idx = next(
                sgkf_temp.split(
                    temp_X,
                    temp_y,
                    temp_groups,
                )
            )

        except Exception as e:
            print(
                f"WARNING: Temp stratified split failed ({e}). "
                "Falling back to GroupShuffleSplit."
            )

            gss_temp = GroupShuffleSplit(
                n_splits=1,
                train_size=val_fraction_of_temp,
                random_state=random_state,
            )

            val_sub_idx, test_sub_idx = next(
                gss_temp.split(
                    temp_X,
                    temp_y,
                    temp_groups,
                )
            )

    else:
        gss_temp = GroupShuffleSplit(
            n_splits=1,
            train_size=val_fraction_of_temp,
            random_state=random_state,
        )

        val_sub_idx, test_sub_idx = next(
            gss_temp.split(
                temp_X,
                temp_y,
                temp_groups,
            )
        )

    # Convert back to original dataframe indices
    val_idx = temp_df.iloc[val_sub_idx].index.values
    test_idx = temp_df.iloc[test_sub_idx].index.values

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------

    train_groups = set(df.iloc[train_idx]["__group__"])
    val_groups = set(df.iloc[val_idx]["__group__"])
    test_groups = set(df.iloc[test_idx]["__group__"])

    assert len(train_groups & val_groups) == 0
    assert len(train_groups & test_groups) == 0
    assert len(val_groups & test_groups) == 0

    if verbose:
        print("\nSplit Summary")
        print("=" * 60)

        summarize_split(df, train_idx, label_col, "TRAIN")
        summarize_split(df, val_idx, label_col, "VALIDATION")
        summarize_split(df, test_idx, label_col, "TEST")

        print("\nLeakage Check")
        print("-" * 60)
        print("Train ∩ Val:", len(train_groups & val_groups))
        print("Train ∩ Test:", len(train_groups & test_groups))
        print("Val ∩ Test:", len(val_groups & test_groups))

    # Remove helper column
    df.drop(columns="__group__", inplace=True)

    return train_idx, val_idx, test_idx


def make_pure_vs_mixture_split(
    df: pd.DataFrame,
    mixture_col: str = "mixture_id",
):
    """
    Specialized split for thesis mixture experiments.

    Strategy:
    ---------
    - Train on pure samples
    - Test on mixture samples

    Returns:
    --------
    train_idx, test_idx
    """

    if mixture_col not in df.columns:
        raise ValueError(f"{mixture_col} not found.")

    pure_mask = df[mixture_col].isna()
    mixture_mask = ~pure_mask

    train_idx = df[pure_mask].index.values
    test_idx = df[mixture_mask].index.values

    print("\nPure vs Mixture Split")
    print("=" * 60)

    print(f"Pure samples (train): {len(train_idx)}")
    print(f"Mixture samples (test): {len(test_idx)}")

    return train_idx, test_idx


if __name__ == "__main__":

    # Example usage
    import pandas as pd

    df = pd.read_parquet(
        "data/processed/bacterial_samples.parquet"
    )

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col="label",
        group_col="raw_file",
        stratify=True,
    )

    print("\nDone.")