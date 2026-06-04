"""
Functions for discovering, parsing, and loading Rapid-E raw data.

Expected folder structure:

data/raw/
├── bacterial_samples/
│   ├── Micrococcus_luteus/
│   │   ├── D_000011209_202107081015.raw
│   │   └── ...
│   ├── Bacillus_endophyticus/
│   │   ├── D_000011255_202107081101.raw
│   │   └── ...
│   └── Ringer_control/
│       ├── D_000010930_202107061122.raw
│       └── ...
└── fluorophores/
    ├── NADH/
    │   ├── D_000012000_202107081300.raw
    │   └── ...
    └── Riboflavin/
        ├── D_000012001_202107081315.raw
        └── ...

Outputs:

data/processed/
├── bacterial_samples.parquet
└── fluorophores.parquet

data/metadata/
├── raw_file_metadata.parquet
└── parse_metadata.parquet
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lif_thesis.data.parsers import parse_raw_file


RAW_ROOT = Path("data/raw")
BACTERIAL_DIR = RAW_ROOT / "bacterial_samples"
FLUOROPHORE_DIR = RAW_ROOT / "fluorophores"

PROCESSED_DIR = Path("data/processed")
METADATA_DIR = Path("data/metadata")


def sha256_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA256 hash for a raw file.

    This supports reproducibility and helps detect accidental file changes.
    """
    path = Path(path)

    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def parse_filename_metadata(path: str | Path) -> dict[str, Any]:
    """
    Parse Rapid-E filenames like:

    D_000011209_202107081015.raw

    Returns:
    - file_prefix
    - file_id
    - file_timestamp
    - filename_parse_ok
    """
    path = Path(path)

    match = re.match(
        r"(?P<prefix>[A-Z])_(?P<file_id>\d+)_(?P<timestamp>\d{12})\.(raw|zip)$",
        path.name,
    )

    if not match:
        return {
            "file_prefix": None,
            "file_id": None,
            "file_timestamp": None,
            "filename_parse_ok": False,
        }

    return {
        "file_prefix": match.group("prefix"),
        "file_id": match.group("file_id"),
        "file_timestamp": datetime.strptime(
            match.group("timestamp"),
            "%Y%m%d%H%M",
        ),
        "filename_parse_ok": True,
    }


def infer_labels_from_folder(
    path: str | Path,
    sample_type: str,
) -> dict[str, Any]:
    """
    Infer labels from folder names.

    For bacterial samples:
        data/raw/bacterial_samples/Micrococcus_luteus/file.raw
        -> species = Micrococcus_luteus
        -> label = Micrococcus_luteus
        -> class_label = bacteria

    For controls:
        data/raw/bacterial_samples/Ringer_control/file.raw
        -> species = Ringer_control
        -> label = Ringer_control
        -> class_label = non_bacteria

    For fluorophores:
        data/raw/fluorophores/NADH/file.raw
        -> fluorophore = NADH
        -> label = NADH
        -> class_label = fluorophore
    """
    path = Path(path)
    folder_label = path.parent.name

    info = {
        "sample_type": sample_type,
        "label": folder_label,
        "species": None,
        "fluorophore": None,
        "class_label": None,
        "condition": None,
        "mixture_id": None,
        "mixture_fraction_a": None,
        "mixture_fraction_b": None,
    }

    lower_label = folder_label.lower()

    if sample_type == "bacterial_sample":
        info["species"] = folder_label
        info["label"] = folder_label

        if any(term in lower_label for term in ["control", "ringer", "blank", "negative"]):
            info["class_label"] = "non_bacteria"
            info["condition"] = "control"
        elif lower_label.startswith("mixture"):
            info["class_label"] = "bacteria"
            info["condition"] = "mixture"
            info["mixture_id"] = folder_label
        else:
            info["class_label"] = "bacteria"
            info["condition"] = "pure"

    elif sample_type == "fluorophore":
        info["fluorophore"] = folder_label
        info["label"] = folder_label
        info["class_label"] = "fluorophore"
        info["condition"] = "fluorophore_standard"

    return info


def discover_raw_files() -> pd.DataFrame:
    """
    Discover all .raw and .zip files in the expected raw data folders.

    Returns one row per raw or zipped raw file.
    """
    records: list[dict[str, Any]] = []

    sources = {
        "bacterial_sample": BACTERIAL_DIR,
        "fluorophore": FLUOROPHORE_DIR,
    }

    for sample_type, folder in sources.items():
        if not folder.exists():
            print(f"Warning: folder does not exist: {folder}")
            continue

        files = sorted(
            list(folder.rglob("*.raw")) + list(folder.rglob("*.zip"))
        )

        for path in files:
            path = Path(path)

            record = {
                "raw_path": str(path),
                "filename": path.name,
                "parent_folder": path.parent.name,
                "relative_path": str(path.relative_to(RAW_ROOT)),
                "file_suffix": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_hash(path),
            }

            record.update(parse_filename_metadata(path))
            record.update(infer_labels_from_folder(path, sample_type))

            records.append(record)

    return pd.DataFrame(records)


def _apply_optional_manifest(discovered: pd.DataFrame) -> pd.DataFrame:
    """
    Optional override using data/metadata/sample_manifest.csv.

    This is useful if folder labels are not sufficient or if you need to
    define specific metadata such as strain, run_id, prep_id, mixture ratios,
    or corrected labels.

    Expected merge key:
        filename

    Example columns:
        filename,species,label,class_label,condition,run_id,prep_id,mixture_id
    """
    manifest_path = METADATA_DIR / "sample_manifest.csv"

    if not manifest_path.exists():
        return discovered

    manifest = pd.read_csv(manifest_path)

    if "filename" not in manifest.columns:
        raise ValueError(
            f"{manifest_path} must contain a 'filename' column."
        )

    merged = discovered.merge(
        manifest,
        on="filename",
        how="left",
        suffixes=("", "_manifest"),
    )

    for col in manifest.columns:
        if col == "filename":
            continue

        manifest_col = f"{col}_manifest"

        if manifest_col in merged.columns:
            merged[col] = merged[manifest_col].combine_first(merged.get(col))
            merged = merged.drop(columns=[manifest_col])

    return merged


def process_raw_dataset(
    keep_thresholds: bool = False,
    extra_params: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Parse all discovered Rapid-E raw files and save particle-level parquet files.

    Saves:
    - data/metadata/raw_file_metadata.parquet
    - data/metadata/parse_metadata.parquet
    - data/processed/bacterial_samples.parquet
    - data/processed/fluorophores.parquet
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    discovered = discover_raw_files()
    discovered = _apply_optional_manifest(discovered)

    if discovered.empty:
        raise FileNotFoundError(
            "No .raw or .zip files found in "
            "data/raw/bacterial_samples or data/raw/fluorophores."
        )

    metadata_path = METADATA_DIR / "raw_file_metadata.parquet"
    discovered.to_parquet(metadata_path, index=False)
    print(f"Saved file metadata: {metadata_path}")

    all_particles: list[pd.DataFrame] = []
    parse_metadata_rows: list[dict[str, Any]] = []

    for _, file_row in discovered.iterrows():
        raw_path = Path(file_row["raw_path"])

        print(f"Parsing: {raw_path}")

        parsed = parse_raw_file(
            raw_path,
            keep_thresholds=keep_thresholds,
            extra_params=extra_params,
        )

        particles = parsed.particle_data.copy()

        for col in discovered.columns:
            if col not in particles.columns:
                particles[col] = file_row[col]

        all_particles.append(particles)

        parse_metadata = parsed.file_metadata.copy()

        for col in [
            "filename",
            "sample_type",
            "label",
            "species",
            "fluorophore",
            "class_label",
            "condition",
            "mixture_id",
            "mixture_fraction_a",
            "mixture_fraction_b",
            "sha256",
            "relative_path",
        ]:
            if col in file_row:
                parse_metadata[col] = file_row[col]

        parse_metadata_rows.append(parse_metadata)

    full_particles = pd.concat(all_particles, ignore_index=True)
    parse_metadata_df = pd.DataFrame(parse_metadata_rows)

    parse_metadata_path = METADATA_DIR / "parse_metadata.parquet"
    parse_metadata_df.to_parquet(parse_metadata_path, index=False)
    print(f"Saved parse metadata: {parse_metadata_path}")

    for sample_type, group in full_particles.groupby("sample_type"):
        if sample_type == "bacterial_sample":
            output_path = PROCESSED_DIR / "bacterial_samples.parquet"
        elif sample_type == "fluorophore":
            output_path = PROCESSED_DIR / "fluorophores.parquet"
        else:
            output_path = PROCESSED_DIR / f"{sample_type}.parquet"

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} already exists and overwrite=False."
            )

        group.to_parquet(output_path, index=False)
        print(f"Saved processed particles: {output_path}")


def load_processed_bacterial_samples() -> pd.DataFrame:
    """
    Load parsed bacterial sample particle data.
    """
    path = PROCESSED_DIR / "bacterial_samples.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/prepare_data.py first."
        )

    return pd.read_parquet(path)


def load_processed_fluorophores() -> pd.DataFrame:
    """
    Load parsed fluorophore particle data.
    """
    path = PROCESSED_DIR / "fluorophores.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/prepare_data.py first."
        )

    return pd.read_parquet(path)


if __name__ == "__main__":
    process_raw_dataset()