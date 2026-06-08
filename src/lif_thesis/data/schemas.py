## Defines expected column names, data formats and valuidation

"""
Schemas and validation models for Rapid-E particle data.

These schemas define the expected structure of particle-level records
throughout the pipeline.

Used by:
    - loaders
    - preprocessing
    - inference
    - API
    - dashboard
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------
# Raw Particle Schema
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class RapidEDimensions:
    """Canonical Rapid-E feature dimensions used across the project."""

    # Fluorescence spectra
    N_SPECTRAL_ACQUISITIONS: int = 8
    N_WAVELENGTHS: int = 32

    # Fluorescence lifetime
    N_LIFETIME_CHANNELS: int = 4
    N_LIFETIME_BINS: int = 64

    # Scattering images
    N_SCATTERING_ANGLES: int = 24
    SCATTERING_TARGET_ACQUISITIONS: int = 60

    @property
    def spectra_features(self) -> int:
        return self.N_SPECTRAL_ACQUISITIONS * self.N_WAVELENGTHS

    @property
    def lifetime_features(self) -> int:
        return self.N_LIFETIME_CHANNELS * self.N_LIFETIME_BINS

    @property
    def scattering_features(self) -> int:
        return self.SCATTERING_TARGET_ACQUISITIONS * self.N_SCATTERING_ANGLES


RAPIDE_DIMS = RapidEDimensions()

class ParticleRecord(BaseModel):
    """
    Raw particle record.

    This represents a single particle detected by the Rapid-E instrument.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    particle_id: str | None = None
    timestamp: str | None = None

    fluorescence_spectra: list[float] | list[list[float]] | None = None

    fluorescence_lifetime: list[float] | list[list[float]] | None = None

    scattering: list[float] | list[list[float]] | None = None

    particle_size: float | None = None

    sample_id: str | None = None
    experiment_id: str | None = None


# ---------------------------------------------------------------------
# Prediction Schema
# ---------------------------------------------------------------------


class ParticlePrediction(BaseModel):
    """
    Prediction output for one particle.
    """

    particle_id: str | None = None

    predicted_label: str

    prediction_confidence: float

    is_unknown: bool = False

    probabilities: dict[str, float] = {}


# ---------------------------------------------------------------------
# Batch Metadata
# ---------------------------------------------------------------------


class BatchMetadata(BaseModel):
    """
    Metadata for a processed particle batch.
    """

    batch_id: str | None = None

    source_file: str | None = None

    processed_at: str | None = None

    n_particles_total: int = 0

    n_particles_eligible: int = 0

    n_particles_rejected: int = 0

    fluorescence_threshold: float = 2000.0


# ---------------------------------------------------------------------
# Internal DataFrame Schema
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleColumns:
    """
    Canonical dataframe column names.

    These names should be used throughout the codebase.
    """

    PARTICLE_ID: str = "particle_id"

    TIMESTAMP: str = "timestamp"

    FLUORESCENCE_SPECTRA: str = "fluorescence_spectra"

    FLUORESCENCE_LIFETIME: str = "fluorescence_lifetime"

    SCATTERING: str = "scattering"

    PEAK_FLUORESCENCE: str = "peak_fluorescence"

    PARTICLE_SIZE: str = "particle_size"

    SAMPLE_ID: str = "sample_id"

    EXPERIMENT_ID: str = "experiment_id"

    PREDICTED_LABEL: str = "predicted_label"

    PREDICTION_CONFIDENCE: str = "prediction_confidence"

    IS_UNKNOWN: str = "is_unknown"


COLUMNS = ParticleColumns()


# ---------------------------------------------------------------------
# Expected Input Schema
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RapidESchema:
    """
    Required and optional columns for particle-level dataframes.
    """

    required_columns: tuple[str, ...] = (
        "fluorescence_spectra",
        "fluorescence_lifetime",
        "scattering",
    )

    optional_columns: tuple[str, ...] = (
        "particle_id",
        "timestamp",
        "particle_size",
        "sample_id",
        "experiment_id",
    )


# ---------------------------------------------------------------------
# Validation Utilities
# ---------------------------------------------------------------------


def validate_particle_dataframe(
    df: pd.DataFrame,
    schema: RapidESchema | None = None,
) -> None:
    """
    Validate a particle-level dataframe.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    schema = schema or RapidESchema()

    missing = [
        column
        for column in schema.required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required particle columns: {missing}"
        )


def validate_particle_record(
    record: dict[str, Any],
) -> ParticleRecord:
    """
    Validate a single particle dictionary.

    Returns
    -------
    ParticleRecord
        Validated particle object.
    """
    return ParticleRecord(**record)


# ---------------------------------------------------------------------
# Array Helpers
# ---------------------------------------------------------------------


def is_array_like(value: Any) -> bool:
    """
    Determine whether a value behaves like an array.
    """
    return isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
        ),
    )


def has_fluorescence_data(
    record: dict[str, Any],
) -> bool:
    """
    Check whether a particle contains fluorescence information.
    """
    spectra = record.get("fluorescence_spectra")
    lifetime = record.get("fluorescence_lifetime")

    return (
        spectra is not None
        or lifetime is not None
    )


__all__ = [
    "ParticleRecord",
    "ParticlePrediction",
    "BatchMetadata",
    "ParticleColumns",
    "RapidESchema",
    "COLUMNS",
    "validate_particle_dataframe",
    "validate_particle_record",
    "is_array_like",
    "has_fluorescence_data",
    "RapidEDimensions",
    "RAPIDE_DIMS",
]