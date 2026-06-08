## Connects file watcher ->> preprocessing >> prediction
"""
Real-time stream processor for Rapid-E particle files.

This module connects the real-time pipeline:

    file_watcher -> loader -> predictor -> prediction outputs

It is designed to work with both:
    - data/realtime_mock/ for thesis demos
    - a live Rapid-E output directory

Expected input files:
    - .csv
    - .parquet
    - .json
    - .jsonl

For raw Rapid-E formats, add the format-specific logic to
`load_particle_file()`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from lif_thesis.data.file_watcher import watch_directory
from lif_thesis.models.predict import (
    ParticlePredictor,
    PredictionConfig,
    summarize_predictions,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamProcessorConfig:
    """Configuration for real-time stream processing."""

    input_dir: Path
    output_dir: Path = Path("results/realtime")
    file_patterns: tuple[str, ...] = ("*.csv", "*.parquet", "*.json", "*.jsonl")
    recursive: bool = False
    include_existing: bool = False
    poll_interval: float = 1.0
    stable_seconds: float = 2.0

    save_particle_predictions: bool = True
    save_minute_summaries: bool = True
    append_outputs: bool = True


def ensure_output_dir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_particle_file(path: Path) -> pd.DataFrame:
    """
    Load one particle-level file.

    Parameters
    ----------
    path:
        Input file path.

    Returns
    -------
    pd.DataFrame
        Particle-level dataframe.

    Notes
    -----
    This assumes the file has already been parsed into a particle-level table.
    If your Rapid-E files are in a custom raw format, connect this function to
    `lif_thesis.data.parsers` or `lif_thesis.data.loaders`.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".json":
        return pd.read_json(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported input file type: {path}")


def add_stream_metadata(df: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    """
    Add metadata columns used by the real-time pipeline.
    """
    output = df.copy()

    output["source_file"] = source_file.name
    output["source_path"] = str(source_file)
    output["processed_at"] = pd.Timestamp.utcnow().isoformat()

    if "event_time" not in output.columns:
        output["event_time"] = output["processed_at"]

    return output


def append_or_write_csv(
    df: pd.DataFrame,
    path: Path,
    append: bool = True,
) -> None:
    """
    Write dataframe to CSV, optionally appending if file already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        df.to_csv(path, mode="a", index=False, header=False)
    else:
        df.to_csv(path, index=False)


def make_file_summary(
    predictions: pd.DataFrame,
    source_file: Path,
) -> pd.DataFrame:
    """
    Create a file-level summary of predictions.
    """
    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "source_file": source_file.name,
                    "processed_at": pd.Timestamp.utcnow().isoformat(),
                    "n_particles_predicted": 0,
                    "n_unknown": 0,
                    "mean_confidence": None,
                    "top_label": None,
                    "top_label_count": 0,
                    "top_label_proportion": 0.0,
                }
            ]
        )

    label_col = "predicted_label"
    confidence_col = "prediction_confidence"

    label_counts = predictions[label_col].value_counts(dropna=False)
    top_label = str(label_counts.index[0])
    top_count = int(label_counts.iloc[0])
    total = int(len(predictions))

    return pd.DataFrame(
        [
            {
                "source_file": source_file.name,
                "processed_at": pd.Timestamp.utcnow().isoformat(),
                "n_particles_predicted": total,
                "n_unknown": int(predictions["is_unknown"].sum())
                if "is_unknown" in predictions.columns
                else None,
                "mean_confidence": float(predictions[confidence_col].mean())
                if confidence_col in predictions.columns
                else None,
                "top_label": top_label,
                "top_label_count": top_count,
                "top_label_proportion": top_count / total if total else 0.0,
            }
        ]
    )


def save_prediction_outputs(
    predictions: pd.DataFrame,
    source_file: Path,
    output_dir: Path,
    append: bool = True,
) -> None:
    """
    Save particle-level predictions and summaries.
    """
    ensure_output_dir(output_dir)

    if not predictions.empty:
        predictions_to_save = predictions.copy()

        # Convert array-like columns to strings so CSV writing does not break.
        for col in predictions_to_save.columns:
            if predictions_to_save[col].apply(lambda x: isinstance(x, (list, tuple))).any():
                predictions_to_save[col] = predictions_to_save[col].apply(json.dumps)

        particle_path = output_dir / "predictions.csv"
        append_or_write_csv(predictions_to_save, particle_path, append=append)

    file_summary = make_file_summary(predictions, source_file)
    append_or_write_csv(
        file_summary,
        output_dir / "file_summary.csv",
        append=append,
    )

    label_summary = summarize_predictions(predictions)
    label_summary["source_file"] = source_file.name
    label_summary["processed_at"] = pd.Timestamp.utcnow().isoformat()

    append_or_write_csv(
        label_summary,
        output_dir / "label_summary.csv",
        append=append,
    )


def process_file(
    path: Path,
    predictor: ParticlePredictor,
    config: StreamProcessorConfig,
) -> pd.DataFrame:
    """
    Process one newly detected particle file.

    Steps:
        1. Load particle file
        2. Add stream metadata
        3. Predict labels
        4. Save outputs
    """
    logger.info("Processing file: %s", path)

    particles = load_particle_file(path)
    particles = add_stream_metadata(particles, path)

    if particles.empty:
        logger.warning("File contains no particles: %s", path)
        predictions = pd.DataFrame()
    else:
        predictions = predictor.predict_particles(particles)

    if config.save_particle_predictions or config.save_minute_summaries:
        save_prediction_outputs(
            predictions=predictions,
            source_file=path,
            output_dir=config.output_dir,
            append=config.append_outputs,
        )

    logger.info("Finished processing %s with %d predictions", path.name, len(predictions))

    return predictions


class StreamProcessor:
    """
    Real-time stream processor.

    Watches a directory and processes each new stable file exactly once.
    """

    def __init__(
        self,
        stream_config: StreamProcessorConfig,
        prediction_config: PredictionConfig,
    ) -> None:
        self.stream_config = stream_config
        self.predictor = ParticlePredictor(prediction_config)
        ensure_output_dir(stream_config.output_dir)

    def run_once(self, files: Iterable[Path]) -> list[pd.DataFrame]:
        """
        Process a finite list of files.

        Useful for tests and debugging.
        """
        outputs: list[pd.DataFrame] = []

        for path in files:
            try:
                predictions = process_file(path, self.predictor, self.stream_config)
                outputs.append(predictions)
            except Exception:
                logger.exception("Failed to process file: %s", path)

        return outputs

    def run_forever(self) -> None:
        """
        Watch input directory continuously and process new files.
        """
        logger.info("Starting stream processor")
        logger.info("Input directory: %s", self.stream_config.input_dir)
        logger.info("Output directory: %s", self.stream_config.output_dir)

        for path in watch_directory(
            directory=self.stream_config.input_dir,
            patterns=self.stream_config.file_patterns,
            poll_interval=self.stream_config.poll_interval,
            stable_seconds=self.stream_config.stable_seconds,
            recursive=self.stream_config.recursive,
            include_existing=self.stream_config.include_existing,
        ):
            try:
                process_file(path, self.predictor, self.stream_config)
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Failed to process file: %s", path)

            time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-time Rapid-E stream processing."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/realtime_mock"),
        help="Directory containing incoming Rapid-E particle files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/realtime"),
        help="Directory where prediction outputs are saved.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model artifact.",
    )

    parser.add_argument(
        "--model-type",
        choices=["sklearn", "torch"],
        default="sklearn",
        help="Model artifact type.",
    )

    parser.add_argument(
        "--label-mapping-path",
        type=Path,
        default=None,
        help="Optional JSON mapping from class index to label.",
    )

    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=None,
        help="Optional feature column list used during training.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence required before assigning a non-unknown label.",
    )

    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process files already present in the input directory.",
    )

    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help="File pattern to watch. Can be repeated.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    stream_config = StreamProcessorConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_patterns=tuple(args.pattern)
        if args.pattern
        else ("*.csv", "*.parquet", "*.json", "*.jsonl"),
        include_existing=args.include_existing,
    )

    prediction_config = PredictionConfig(
        model_path=args.model_path,
        model_type=args.model_type,
        label_mapping_path=args.label_mapping_path,
        feature_columns_path=args.feature_columns_path,
        confidence_threshold=args.confidence_threshold,
    )

    processor = StreamProcessor(
        stream_config=stream_config,
        prediction_config=prediction_config,
    )

    processor.run_forever()


if __name__ == "__main__":
    main()