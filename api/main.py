"""
FastAPI application for real-time Rapid-E bioaerosol classification.

Run locally:
    uvicorn api.main:app --reload

Run in Docker:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lif_thesis.realtime.inference_engine import (
    InferenceEngine,
    InferenceEngineConfig,
)
from lif_thesis.realtime.prediction_store import (
    PredictionStore,
    PredictionStoreConfig,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/trained/rf_species.joblib"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "sklearn")
LABEL_MAPPING_PATH = os.getenv("LABEL_MAPPING_PATH")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH")

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
FLUORESCENCE_THRESHOLD = float(os.getenv("FLUORESCENCE_THRESHOLD", "2000"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "results/realtime"))
STORE_BACKEND = os.getenv("STORE_BACKEND", "both")


# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------

app = FastAPI(
    title="LIF Thesis Rapid-E Bioaerosol Classifier",
    description=(
        "API for particle-level bacterial bioaerosol classification "
        "using Rapid-E fluorescence, lifetime, and scattering data."
    ),
    version="0.1.0",
)


engine: InferenceEngine | None = None
store: PredictionStore | None = None


# ---------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------

class ParticleRequest(BaseModel):
    """
    Single particle request.

    The expected fields should match your preprocessing pipeline:
        - fluorescence_spectra
        - fluorescence_lifetime
        - scattering

    These are kept flexible because Rapid-E parsing may produce nested arrays
    or flattened values depending on the stage of the pipeline.
    """

    particle: dict[str, Any] = Field(
        ...,
        description="Single particle record as a JSON object.",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""

    particles: list[dict[str, Any]] = Field(
        ...,
        description="List of particle records.",
    )
    batch_id: str | None = Field(
        default=None,
        description="Optional batch identifier.",
    )
    source_file: str | None = Field(
        default=None,
        description="Optional source filename.",
    )
    save: bool = Field(
        default=True,
        description="Whether to save predictions to CSV/SQLite.",
    )


class FilePredictionRequest(BaseModel):
    """Prediction request from an already parsed particle file."""

    file_path: str = Field(
        ...,
        description="Path to CSV, parquet, JSON, or JSONL particle file.",
    )
    batch_id: str | None = None
    save: bool = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    output_dir: str


# ---------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------

@app.on_event("startup")
def startup_event() -> None:
    """Initialize model inference engine and prediction store."""
    global engine, store

    if not MODEL_PATH.exists():
        # Do not crash the API during early development.
        # Health endpoint will report model_loaded=False.
        engine = None
    else:
        engine = InferenceEngine(
            InferenceEngineConfig(
                model_path=MODEL_PATH,
                model_type=MODEL_TYPE,
                label_mapping_path=Path(LABEL_MAPPING_PATH)
                if LABEL_MAPPING_PATH
                else None,
                feature_columns_path=Path(FEATURE_COLUMNS_PATH)
                if FEATURE_COLUMNS_PATH
                else None,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                fluorescence_threshold=FLUORESCENCE_THRESHOLD,
            )
        )

    store = PredictionStore(
        PredictionStoreConfig(
            output_dir=OUTPUT_DIR,
            backend=STORE_BACKEND,  # type: ignore[arg-type]
        )
    )


def require_engine() -> InferenceEngine:
    """Return initialized engine or raise HTTP error."""
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded. Expected model at: {MODEL_PATH}. "
                "Set MODEL_PATH or place a trained model in models/trained/."
            ),
        )

    return engine


def require_store() -> PredictionStore:
    """Return initialized prediction store or raise HTTP error."""
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction store not initialized.",
        )

    return store


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def load_particle_file(path: Path) -> pd.DataFrame:
    """Load an already parsed particle-level file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".json":
        return pd.read_json(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    raise ValueError(f"Unsupported file type: {path.suffix}")


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert dataframe to JSON-safe records.

    This is intentionally conservative for API responses.
    """
    if df.empty:
        return []

    safe = df.copy()

    for col in safe.columns:
        safe[col] = safe[col].map(
            lambda x: x.tolist() if hasattr(x, "tolist") else x
        )

    return safe.to_dict(orient="records")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/", tags=["Status"])
def root() -> dict[str, str]:
    return {
        "message": "LIF Thesis Rapid-E Bioaerosol Classifier API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None,
        model_path=str(MODEL_PATH),
        output_dir=str(OUTPUT_DIR),
    )


@app.post("/predict", tags=["Prediction"])
def predict_one(request: ParticleRequest) -> dict[str, Any]:
    """Predict a single particle."""
    active_engine = require_engine()
    return active_engine.predict_one(request.particle)


@app.post("/batch_predict", tags=["Prediction"])
def batch_predict(request: BatchPredictionRequest) -> dict[str, Any]:
    """Predict a batch of particles."""
    active_engine = require_engine()

    batch_id = request.batch_id or str(uuid.uuid4())

    particles = pd.DataFrame(request.particles)

    output = active_engine.predict_batch(
        particles=particles,
        batch_id=batch_id,
        source_file=request.source_file,
    )

    if request.save:
        active_store = require_store()
        active_store.save_batch(
            predictions=output.predictions,
            source_file=request.source_file,
            batch_id=batch_id,
        )

    return {
        "batch_id": batch_id,
        "metadata": output.metadata,
        "summary": dataframe_to_records(output.summary),
        "predictions": dataframe_to_records(output.predictions),
    }


@app.post("/predict_file", tags=["Prediction"])
def predict_file(request: FilePredictionRequest) -> dict[str, Any]:
    """
    Predict all particles from an already parsed file.

    This is useful for replaying files from data/realtime_mock.
    """
    active_engine = require_engine()

    path = Path(request.file_path)
    batch_id = request.batch_id or path.stem

    try:
        particles = load_particle_file(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output = active_engine.predict_batch(
        particles=particles,
        batch_id=batch_id,
        source_file=path.name,
    )

    if request.save:
        active_store = require_store()
        active_store.save_batch(
            predictions=output.predictions,
            source_file=path.name,
            batch_id=batch_id,
        )

    return {
        "batch_id": batch_id,
        "metadata": output.metadata,
        "summary": dataframe_to_records(output.summary),
        "predictions": dataframe_to_records(output.predictions),
    }


@app.get("/latest", tags=["Storage"])
def latest_predictions(limit: int = 100) -> dict[str, Any]:
    """Return latest stored predictions from SQLite."""
    active_store = require_store()

    try:
        predictions = active_store.read_predictions(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "limit": limit,
        "predictions": dataframe_to_records(predictions),
    }


@app.get("/summary", tags=["Storage"])
def latest_summary(limit: int = 100) -> dict[str, Any]:
    """Return latest stored label summaries from SQLite."""
    active_store = require_store()

    try:
        label_summary = active_store.read_label_summary(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "limit": limit,
        "summary": dataframe_to_records(label_summary),
    }


@app.get("/composition", tags=["Analysis"])
def composition(limit: int = 1000) -> dict[str, Any]:
    """Estimate composition from recent stored predictions."""
    active_engine = require_engine()
    active_store = require_store()

    try:
        predictions = active_store.read_predictions(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    composition_df = active_engine.estimate_composition(predictions)

    return {
        "limit": limit,
        "composition": dataframe_to_records(composition_df),
    }