##1D CNN for direct spectra analysis

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lif_thesis.data.splits import make_group_split
from lif_thesis.evaluation.metrics import compute_classification_metrics


class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # CNN expects: channels x sequence_length
        return self.X[idx].unsqueeze(0), self.y[idx]


class BaselineCNN1D(nn.Module):
    """
    Simple 1D CNN for fluorescence spectra.

    Input shape:
        batch_size x 1 x n_spectral_bins
    """

    def __init__(self, input_length: int, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_spectra_matrix(
    df: pd.DataFrame,
    spectra_col: str = "spectrometer",
) -> np.ndarray:
    X = np.stack(df[spectra_col].apply(lambda x: np.asarray(x)).values)

    if X.ndim != 2:
        raise ValueError(f"Expected spectra matrix with shape (n, bins), got {X.shape}")

    return X


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_samples += len(X_batch)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_y = []
    all_pred = []
    all_proba = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        proba = torch.softmax(logits, dim=1)
        pred = proba.argmax(dim=1)

        total_loss += loss.item() * len(X_batch)
        total_samples += len(X_batch)

        all_y.append(y_batch.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return {
        "loss": total_loss / total_samples,
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_pred),
        "y_proba": np.concatenate(all_proba),
    }


def run_baseline_cnn_experiment(
    df: pd.DataFrame,
    label_col: str = "label",
    group_col: str = "raw_file",
    spectra_col: str = "spectrometer",
    output_dir: str | Path = "results/baseline_cnn1d",
    batch_size: int = 64,
    max_epochs: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    random_state: int = 42,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df[spectra_col].notna()
    ].reset_index(drop=True)

    print(f"Using {len(df)} rows.")
    print(f"Labels: {df[label_col].value_counts().to_dict()}")
    print(f"Groups: {df[group_col].nunique()}")

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

    # Fit scaler ONLY on train data to prevent leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    train_loader = DataLoader(
        SpectraDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        SpectraDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        SpectraDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BaselineCNN1D(
        input_length=X.shape[1],
        n_classes=len(label_encoder.classes_),
    ).to(device)

    # Class weighting helps if bacterial classes are imbalanced
    class_counts = np.bincount(y_train)
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_eval = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = compute_classification_metrics(
            y_true=val_eval["y_true"],
            y_pred=val_eval["y_pred"],
            y_proba=val_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_eval["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # Load best model before final evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    train_eval = evaluate_model(model, train_loader, criterion, device)
    val_eval = evaluate_model(model, val_loader, criterion, device)
    test_eval = evaluate_model(model, test_loader, criterion, device)

    metrics = {
        "train": compute_classification_metrics(
            train_eval["y_true"],
            train_eval["y_pred"],
            train_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "val": compute_classification_metrics(
            val_eval["y_true"],
            val_eval["y_pred"],
            val_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "test": compute_classification_metrics(
            test_eval["y_true"],
            test_eval["y_pred"],
            test_eval["y_proba"],
            class_names=label_encoder.classes_.tolist(),
        ),
        "best_epoch": best_epoch,
        "label_classes": label_encoder.classes_.tolist(),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    joblib.dump(scaler, output_dir / "scaler.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "val_idx.npy", val_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    print(f"\nSaved CNN outputs to: {output_dir}")

    return model, label_encoder, scaler, metrics