"""
Prediction storage utilities for the real-time Rapid-E pipeline.

Supports:
    - CSV storage for simple thesis demos
    - SQLite storage for dashboard/API use
    - batch-level summaries
    - label composition summaries

Typical outputs:
    results/realtime/predictions.csv
    results/realtime/file_summary.csv
    results/realtime/label_summary.csv
    results/realtime/predictions.sqlite
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

StorageBackend = Literal["csv", "sqlite", "both"]


@dataclass(frozen=True)
class PredictionStoreConfig:
    """Configuration for prediction storage."""

    output_dir: Path = Path("results/realtime")
    sqlite_path: Path | None = None
    backend: StorageBackend = "both"
    append_csv: bool = True

    predictions_csv: str = "predictions.csv"
    file_summary_csv: str = "file_summary.csv"
    label_summary_csv: str = "label_summary.csv"


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def make_json_safe(value: Any) -> Any:
    """
    Convert values to JSON/SQLite safe formats.

    Arrays, lists, and dictionaries are serialized to JSON strings.
    NumPy scalar values are converted to Python scalar values.
    Timestamps are converted to ISO strings.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None

    return value


def make_dataframe_storage_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dataframe values into CSV/SQLite-safe representations.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    for col in out.columns:
        out[col] = out[col].map(make_json_safe)

    return out


def append_or_write_csv(
    df: pd.DataFrame,
    path: Path,
    append: bool = True,
) -> None:
    """
    Append dataframe to CSV if it exists, otherwise create it.
    """
    ensure_dir(path.parent)

    safe_df = make_dataframe_storage_safe(df)

    if append and path.exists():
        safe_df.to_csv(path, mode="a", index=False, header=False)
    else:
        safe_df.to_csv(path, index=False)


def summarize_file(
    predictions: pd.DataFrame,
    source_file: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    """
    Create one-row summary for a processed file or batch.
    """
    processed_at = pd.Timestamp.utcnow().isoformat()

    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "batch_id": batch_id,
                    "source_file": source_file,
                    "processed_at": processed_at,
                    "n_predictions": 0,
                    "n_unknown": 0,
                    "unknown_fraction": None,
                    "mean_confidence": None,
                    "top_label": None,
                    "top_label_count": 0,
                    "top_label_proportion": 0.0,
                }
            ]
        )

    label_col = "predicted_label"
    confidence_col = "prediction_confidence"

    counts = predictions[label_col].value_counts(dropna=False)
    top_label = str(counts.index[0])
    top_count = int(counts.iloc[0])
    total = int(len(predictions))

    n_unknown = (
        int(predictions["is_unknown"].sum())
        if "is_unknown" in predictions.columns
        else None
    )

    mean_confidence = (
        float(predictions[confidence_col].mean())
        if confidence_col in predictions.columns
        else None
    )

    return pd.DataFrame(
        [
            {
                "batch_id": batch_id,
                "source_file": source_file,
                "processed_at": processed_at,
                "n_predictions": total,
                "n_unknown": n_unknown,
                "unknown_fraction": n_unknown / total
                if n_unknown is not None and total
                else None,
                "mean_confidence": mean_confidence,
                "top_label": top_label,
                "top_label_count": top_count,
                "top_label_proportion": top_count / total if total else 0.0,
            }
        ]
    )


def summarize_labels(
    predictions: pd.DataFrame,
    source_file: str | None = None,
    batch_id: str | None = None,
) -> pd.DataFrame:
    """
    Create label-level count and proportion summary.
    """
    processed_at = pd.Timestamp.utcnow().isoformat()

    if predictions.empty or "predicted_label" not in predictions.columns:
        return pd.DataFrame(
            columns=[
                "batch_id",
                "source_file",
                "processed_at",
                "predicted_label",
                "count",
                "proportion",
                "mean_confidence",
            ]
        )

    total = len(predictions)

    summary = (
        predictions.groupby("predicted_label", dropna=False)
        .agg(
            count=("predicted_label", "size"),
            mean_confidence=("prediction_confidence", "mean")
            if "prediction_confidence" in predictions.columns
            else ("predicted_label", "size"),
        )
        .reset_index()
    )

    summary["proportion"] = summary["count"] / total
    summary["batch_id"] = batch_id
    summary["source_file"] = source_file
    summary["processed_at"] = processed_at

    return summary[
        [
            "batch_id",
            "source_file",
            "processed_at",
            "predicted_label",
            "count",
            "proportion",
            "mean_confidence",
        ]
    ].sort_values("count", ascending=False)


class PredictionStore:
    """
    Store live predictions to CSV, SQLite, or both.
    """

    def __init__(self, config: PredictionStoreConfig | None = None) -> None:
        self.config = config or PredictionStoreConfig()
        ensure_dir(self.config.output_dir)

        self.sqlite_path = (
            self.config.sqlite_path
            if self.config.sqlite_path is not None
            else self.config.output_dir / "predictions.sqlite"
        )

        if self.config.backend in ("sqlite", "both"):
            self.initialize_sqlite()

    def connect(self) -> sqlite3.Connection:
        """Open SQLite connection."""
        ensure_dir(self.sqlite_path.parent)
        return sqlite3.connect(self.sqlite_path)

    def initialize_sqlite(self) -> None:
        """
        Initialize SQLite database.

        Tables are created automatically by pandas.to_sql, but this verifies the
        database path exists and can be opened.
        """
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")

        logger.info("SQLite prediction store initialized at %s", self.sqlite_path)

    def save_predictions_csv(
        self,
        predictions: pd.DataFrame,
    ) -> None:
        """Save particle-level predictions to CSV."""
        path = self.config.output_dir / self.config.predictions_csv
        append_or_write_csv(predictions, path, append=self.config.append_csv)

    def save_file_summary_csv(
        self,
        file_summary: pd.DataFrame,
    ) -> None:
        """Save file/batch-level summary to CSV."""
        path = self.config.output_dir / self.config.file_summary_csv
        append_or_write_csv(file_summary, path, append=self.config.append_csv)

    def save_label_summary_csv(
        self,
        label_summary: pd.DataFrame,
    ) -> None:
        """Save label-level summary to CSV."""
        path = self.config.output_dir / self.config.label_summary_csv
        append_or_write_csv(label_summary, path, append=self.config.append_csv)

    def save_dataframe_sqlite(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> None:
        """Append dataframe to SQLite table."""
        if df.empty:
            return

        safe_df = make_dataframe_storage_safe(df)

        with self.connect() as conn:
            safe_df.to_sql(table_name, conn, if_exists="append", index=False)

    def save_batch(
        self,
        predictions: pd.DataFrame,
        source_file: str | None = None,
        batch_id: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Save one batch of predictions.

        Parameters
        ----------
        predictions:
            Particle-level prediction dataframe.
        source_file:
            Optional source file name.
        batch_id:
            Optional batch identifier.

        Returns
        -------
        dict[str, pd.DataFrame]
            Saved dataframes:
                - predictions
                - file_summary
                - label_summary
        """
        predictions = predictions.copy()

        processed_at = pd.Timestamp.utcnow().isoformat()

        if "batch_id" not in predictions.columns:
            predictions["batch_id"] = batch_id

        if "source_file" not in predictions.columns:
            predictions["source_file"] = source_file

        if "stored_at" not in predictions.columns:
            predictions["stored_at"] = processed_at

        file_summary = summarize_file(
            predictions=predictions,
            source_file=source_file,
            batch_id=batch_id,
        )

        label_summary = summarize_labels(
            predictions=predictions,
            source_file=source_file,
            batch_id=batch_id,
        )

        if self.config.backend in ("csv", "both"):
            self.save_predictions_csv(predictions)
            self.save_file_summary_csv(file_summary)
            self.save_label_summary_csv(label_summary)

        if self.config.backend in ("sqlite", "both"):
            self.save_dataframe_sqlite(predictions, "predictions")
            self.save_dataframe_sqlite(file_summary, "file_summary")
            self.save_dataframe_sqlite(label_summary, "label_summary")

        logger.info(
            "Stored prediction batch: batch_id=%s source_file=%s n=%d",
            batch_id,
            source_file,
            len(predictions),
        )

        return {
            "predictions": predictions,
            "file_summary": file_summary,
            "label_summary": label_summary,
        }

    def read_predictions(self, limit: int | None = 1000) -> pd.DataFrame:
        """Read recent predictions from SQLite."""
        query = "SELECT * FROM predictions"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def read_file_summary(self, limit: int | None = 1000) -> pd.DataFrame:
        """Read recent file summaries from SQLite."""
        query = "SELECT * FROM file_summary"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def read_label_summary(self, limit: int | None = 1000) -> pd.DataFrame:
        """Read recent label summaries from SQLite."""
        query = "SELECT * FROM label_summary"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        with self.connect() as conn:
            return pd.read_sql_query(query, conn)


def create_prediction_store(
    output_dir: str | Path = "results/realtime",
    backend: StorageBackend = "both",
) -> PredictionStore:
    """Convenience factory for creating a prediction store."""
    return PredictionStore(
        PredictionStoreConfig(
            output_dir=Path(output_dir),
            backend=backend,
        )
    )


__all__ = [
    "PredictionStoreConfig",
    "PredictionStore",
    "create_prediction_store",
    "summarize_file",
    "summarize_labels",
]