## Implements leakage-free data splits for training, validation and testing

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_split_inputs(
    df: pd.DataFrame,
    label_col: str,
    group_col: str,
):
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
    group_col: str,
    name: str,
):
    subset = df.iloc[indices]

    print(f"\n{name}")
    print("-" * 60)
    print(f"N rows: {len(subset)}")
    print(f"N groups: {subset[group_col].nunique()}")

    print("\nClass distribution:")
    print(subset[label_col].value_counts(normalize=True))

    print("\nGroups per class:")
    print(subset.groupby(label_col)[group_col].nunique())


def _split_groups_for_one_label(
    groups: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    rng: np.random.Generator,
):
    """
    Split unique groups for a single label into train/val/test.

    Ensures at least one validation and one test group when possible.
    """
    groups = np.array(groups)
    rng.shuffle(groups)

    n_groups = len(groups)

    if n_groups < 3:
        raise ValueError(
            "Each label needs at least 3 groups to create train/val/test splits. "
            f"Found only {n_groups} groups."
        )

    n_train = int(round(train_size * n_groups))
    n_val = int(round(val_size * n_groups))
    n_test = n_groups - n_train - n_val

    # Ensure val/test are represented when possible
    if n_val < 1:
        n_val = 1
    if n_test < 1:
        n_test = 1

    # Adjust train if val/test correction made the total too large
    n_train = n_groups - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough groups to split. Got {n_groups} groups."
        )

    train_groups = groups[:n_train]
    val_groups = groups[n_train:n_train + n_val]
    test_groups = groups[n_train + n_val:]

    return train_groups, val_groups, test_groups


def make_group_split(
    df: pd.DataFrame,
    label_col: str = "species",
    group_col: str = "raw_file",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Create leakage-free grouped train/validation/test splits.

    This implementation is designed for datasets like yours where:
    - each raw_file belongs to exactly one species
    - raw_file should not appear in more than one split
    - every species should appear in train, validation, and test

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Integer row positions for df.iloc[...].
    """

    validate_split_inputs(df, label_col, group_col)

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train/val/test sizes must sum to 1. Got {total}"
        )

    df = df.copy().reset_index(drop=True)

    # Check that each group has only one label.
    group_label_counts = df.groupby(group_col)[label_col].nunique()
    invalid_groups = group_label_counts[group_label_counts > 1]

    if len(invalid_groups) > 0:
        raise ValueError(
            "Some groups contain more than one label. "
            "This split assumes each group belongs to exactly one class. "
            f"Invalid groups: {invalid_groups.index.tolist()[:10]}"
        )

    rng = np.random.default_rng(random_state)

    train_groups_all = []
    val_groups_all = []
    test_groups_all = []

    if stratify:
        # Split groups separately within each label.
        for label, label_df in df.groupby(label_col):
            unique_groups = label_df[group_col].unique()

            train_groups, val_groups, test_groups = _split_groups_for_one_label(
                groups=unique_groups,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                rng=rng,
            )

            train_groups_all.extend(train_groups)
            val_groups_all.extend(val_groups)
            test_groups_all.extend(test_groups)

    else:
        # Non-stratified fallback: shuffle all groups globally.
        unique_groups = df[group_col].unique()
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train = int(round(train_size * n_groups))
        n_val = int(round(val_size * n_groups))
        n_test = n_groups - n_train - n_val

        if n_val < 1:
            n_val = 1
        if n_test < 1:
            n_test = 1

        n_train = n_groups - n_val - n_test

        train_groups_all = unique_groups[:n_train]
        val_groups_all = unique_groups[n_train:n_train + n_val]
        test_groups_all = unique_groups[n_train + n_val:]

    train_groups_all = set(train_groups_all)
    val_groups_all = set(val_groups_all)
    test_groups_all = set(test_groups_all)

    # Leakage checks
    assert len(train_groups_all & val_groups_all) == 0
    assert len(train_groups_all & test_groups_all) == 0
    assert len(val_groups_all & test_groups_all) == 0

    train_idx = df.index[df[group_col].isin(train_groups_all)].to_numpy()
    val_idx = df.index[df[group_col].isin(val_groups_all)].to_numpy()
    test_idx = df.index[df[group_col].isin(test_groups_all)].to_numpy()

    if verbose:
        print("\nSplit Summary")
        print("=" * 60)

        summarize_split(df, train_idx, label_col, group_col, "TRAIN")
        summarize_split(df, val_idx, label_col, group_col, "VALIDATION")
        summarize_split(df, test_idx, label_col, group_col, "TEST")

        print("\nLeakage Check")
        print("-" * 60)
        print("Train ∩ Val:", len(train_groups_all & val_groups_all))
        print("Train ∩ Test:", len(train_groups_all & test_groups_all))
        print("Val ∩ Test:", len(val_groups_all & test_groups_all))

    return train_idx, val_idx, test_idx


def make_pure_vs_mixture_split(
    df: pd.DataFrame,
    mixture_col: str = "mixture_id",
):
    """
    Specialized split for thesis mixture experiments.

    Strategy:
    - Train on pure samples
    - Test on mixture samples
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
    df = pd.read_parquet("data/processed/bacterial_samples.parquet")

    train_idx, val_idx, test_idx = make_group_split(
        df,
        label_col="species",
        group_col="raw_file",
        stratify=True,
    )

    print("\nDone.")