"""
Experiment 04: Baseline 1D CNN on fluorescence spectra.

Purpose:
Train a simple 1D CNN to classify bacterial species from particle-level
fluorescence spectra.

Protocol:
- Peak fluorescence threshold > 2000 a.u.
- Grouped raw-file train/validation/test split
- Input: spectrometer vector only
- Output: 5-class species prediction
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from lif_thesis.data.splits import make_group_split


DATA_PATH = Path("data/processed/bacterial_samples.parquet")
OUTPUT_DIR = Path("results/exp04_baseline_cnn")


def to_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.asarray(x)
    return np.asarray(x)


def peak_fluorescence(spectrum) -> float:
    return float(np.max(to_array(spectrum)))


def build_spectra_matrix(df: pd.DataFrame) -> np.ndarray:
    X = np.stack(df["spectrometer"].apply(to_array).values)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D spectra matrix, got {X.shape}")

    return X


class SpectraDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Shape for Conv1D: channels x sequence length
        return self.X[idx].unsqueeze(0), self.y[idx]


class BaselineCNN1D(nn.Module):
    def __init__(self, n_classes: int):
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
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    total_correct = 0
    total_n = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total_n += len(X_batch)

    return {
        "loss": total_loss / total_n,
        "accuracy": total_correct / total_n,
    }


@torch.no_grad()
def evaluate_torch_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    total_n = 0

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
        total_n += len(X_batch)

        all_y.append(y_batch.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    return {
        "loss": total_loss / total_n,
        "y_true": np.concatenate(all_y),
        "y_pred": np.concatenate(all_pred),
        "y_proba": np.concatenate(all_proba),
    }


def compute_metrics(y_true, y_pred, label_encoder):
    labels = np.arange(len(label_encoder.classes_))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_encoder.classes_.astype(str),
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_col = "species"
    group_col = "raw_file"
    fluorescence_threshold = 2000
    random_state = 42

    batch_size = 128
    max_epochs = 50
    patience = 8
    learning_rate = 1e-3
    weight_decay = 1e-4

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    df = df[
        df[label_col].notna()
        & df[group_col].notna()
        & df["spectrometer"].notna()
    ].reset_index(drop=True)

    print(f"Particles before threshold: {len(df)}")

    df["peak_fluorescence"] = df["spectrometer"].apply(peak_fluorescence)
    df = df[df["peak_fluorescence"] > fluorescence_threshold].reset_index(drop=True)

    print(f"Particles after fluorescence > {fluorescence_threshold}: {len(df)}")
    print(df[label_col].value_counts())

    X = build_spectra_matrix(df)

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

    # Scale using training data only
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
    print("Device:", device)

    model = BaselineCNN1D(
        n_classes=len(label_encoder.classes_)
    ).to(device)

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
    best_epoch = 0
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

        val_eval = evaluate_torch_model(
            model,
            val_loader,
            criterion,
            device,
        )

        val_metrics = compute_metrics(
            val_eval["y_true"],
            val_eval["y_pred"],
            label_encoder,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_eval["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_bal_acc={record['val_balanced_accuracy']:.4f} | "
            f"val_macro_f1={record['val_macro_f1']:.4f}"
        )

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(
        torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
    )

    train_eval = evaluate_torch_model(model, train_loader, criterion, device)
    val_eval = evaluate_torch_model(model, val_loader, criterion, device)
    test_eval = evaluate_torch_model(model, test_loader, criterion, device)

    metrics = {
        "experiment": {
            "name": "exp04_baseline_cnn",
            "label_col": label_col,
            "group_col": group_col,
            "input_features": ["spectrometer"],
            "fluorescence_threshold": fluorescence_threshold,
            "split_protocol": "grouped_raw_file_60_train_20_val_20_test",
            "model": "1D CNN",
        },
        "best_epoch": best_epoch,
        "train": compute_metrics(
            train_eval["y_true"],
            train_eval["y_pred"],
            label_encoder,
        ),
        "val": compute_metrics(
            val_eval["y_true"],
            val_eval["y_pred"],
            label_encoder,
        ),
        "test": compute_metrics(
            test_eval["y_true"],
            test_eval["y_pred"],
            label_encoder,
        ),
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False,
    )

    joblib.dump(label_encoder, OUTPUT_DIR / "label_encoder.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "val_idx.npy", val_idx)
    np.save(OUTPUT_DIR / "test_idx.npy", test_idx)

    print(f"\nSaved outputs to: {OUTPUT_DIR}")

    print("\nFinal test performance:")
    print(
        json.dumps(
            {
                "accuracy": metrics["test"]["accuracy"],
                "balanced_accuracy": metrics["test"]["balanced_accuracy"],
                "macro_f1": metrics["test"]["macro_f1"],
                "best_epoch": best_epoch,
            },
            indent=4,
        )
    )


if __name__ == "__main__":
    main()