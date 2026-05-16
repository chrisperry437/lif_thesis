##Functions to load raw data files and convert them to the required format

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
FLUOROPHORE_DIR = RAW_ROOT / "fluorophores"  # keeping your current spelling

PROCESSED_DIR = Path("data/processed")
METADATA_DIR = Path("data/metadata")


def sha256_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def parse_filename_metadata(path: str | Path) -> dict[str, Any]:
    """
    Parse Rapid-E filenames like:
    D_000000096_202107081015.raw

    Returns file_id and file timestamp when possible.
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

    ts = datetime.strptime(match.group("timestamp"), "%Y%m%d%H%M")

    return {
        "file_prefix": match.group("prefix"),
        "file_id": match.group("file_id"),
        "file_timestamp": ts,
        "filename_parse_ok": True,
    }


def infer_label_from_path(path: str | Path, sample_type: str) -> dict[str, Any]:
    """
    Infer labels from folder structure.

    Recommended folder structures:

    data/raw/bacterial_samples/Micrococcus_luteus/D_....raw
    data/raw/bacterial_samples/Bacillus_endophyticus/D_....raw

    data/raw/flourophores/NADH/D_....raw
    data/raw/flourophores/Riboflavin/D_....raw

    For mixtures, you can later use:

    data/raw/bacterial_samples/mixture_M_luteus_75_B_endophyticus_25/D_....raw
    """

    path = Path(path)
    parent = path.parent.name

    label_info = {
        "sample_type": sample_type,
        "label": parent,
        "species": None,
        "fluorophore": None,
        "mixture_id": None,
        "mixture_fraction_a": None,
        "mixture_fraction_b": None,
    }

    if sample_type == "bacterial_sample":
        label_info["species"] = parent

        if parent.lower().startswith("mixture"):
            label_info["mixture_id"] = parent

    elif sample_type == "fluorophore":
        label_info["fluorophore"] = parent

    return label_info


def discover_raw_files() -> pd.DataFrame:
    """
    Find .raw and .zip files from bacterial and fluorophore raw folders.
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

        files = list(folder.rglob("*.raw")) + list(folder.rglob("*.zip"))

        for path in sorted(files):
            record = {
                "raw_path": str(path),
                "filename": path.name,
                "parent_folder": path.parent.name,
                "file_suffix": path.suffix,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256_hash(path),
            }

            record.update(parse_filename_metadata(path))
            record.update(infer_label_from_path(path, sample_type))

            records.append(record)

    return pd.DataFrame(records)


def process_raw_dataset(
    keep_thresholds: bool = False,
    extra_params: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Parse all raw Rapid-E files and save:

    data/processed/bacterial_samples.parquet
    data/processed/fluorophores.parquet
    data/metadata/raw_file_metadata.parquet
    data/metadata/parse_metadata.parquet
    """

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    discovered = discover_raw_files()

    if discovered.empty:
        raise FileNotFoundError(
            "No .raw or .zip files found in "
            "data/raw/bacterial_samples or data/raw/flourophores."
        )

    discovered_path = METADATA_DIR / "raw_file_metadata.parquet"
    discovered.to_parquet(discovered_path, index=False)
    print(f"Saved file metadata: {discovered_path}")

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

        particles = parsed.particle_data

        # Attach file-level metadata and labels to every particle row.
        for col in discovered.columns:
            if col not in particles.columns:
                particles[col] = file_row[col]

        all_particles.append(particles)

        parse_metadata = parsed.file_metadata.copy()
        parse_metadata.update(
            {
                "sample_type": file_row["sample_type"],
                "label": file_row["label"],
                "species": file_row["species"],
                "fluorophore": file_row["fluorophore"],
                "mixture_id": file_row["mixture_id"],
                "sha256": file_row["sha256"],
            }
        )

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
            raise FileExistsError(f"{output_path} already exists and overwrite=False")

        group.to_parquet(output_path, index=False)
        print(f"Saved processed particles: {output_path}")


def load_processed_bacterial_samples() -> pd.DataFrame:
    path = PROCESSED_DIR / "bacterial_samples.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/prepare_data.py first."
        )

    return pd.read_parquet(path)


def load_processed_fluorophores() -> pd.DataFrame:
    path = PROCESSED_DIR / "fluorophores.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/prepare_data.py first."
        )

    return pd.read_parquet(path)


if __name__ == "__main__":
    process_raw_dataset()
