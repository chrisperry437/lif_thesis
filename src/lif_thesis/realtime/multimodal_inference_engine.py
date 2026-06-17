# src/lif_thesis/realtime/multimodal_inference_engine.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from lif_thesis.data.schemas import RAPIDE_DIMS
from lif_thesis.experiments.exp05_multimodal_deep_learning import (
    MultimodalDeepClassifier,
    crop_pad_scattering,
    stack_vector_column,
)


class MultimodalInferenceEngine:
    """
    Real-time inference engine for Exp05 multimodal deep learning model.

    Expected input dataframe columns:
        - spectrometer
        - lifetime
        - scattering_image
        - size
        - time_asymmetry
    """

    def __init__(
        self,
        model_path: Path | str = Path("models/trained/multimodal_species_v1.pt"),
        config_dir: Path | str = Path("models/configs"),
        label_map_path: Path | str = Path("models/label_maps/multimodal_species_v1_labels.json"),
        model_name: str = "multimodal_species_v1",
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.config_dir = Path(config_dir)
        self.label_map_path = Path(label_map_path)
        self.model_name = model_name

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        self.class_names = self._load_class_names()
        self.model = self._load_model()

        self.spec_scaler = joblib.load(
            self.config_dir / f"{self.model_name}_spectrometer_scaler.joblib"
        )
        self.life_scaler = joblib.load(
            self.config_dir / f"{self.model_name}_lifetime_scaler.joblib"
        )
        self.scat_scaler = joblib.load(
            self.config_dir / f"{self.model_name}_scattering_scaler.joblib"
        )
        self.scalar_scaler = joblib.load(
            self.config_dir / f"{self.model_name}_scalar_scaler.joblib"
        )

    def _load_class_names(self) -> list[str]:
        if "class_names" in self.checkpoint:
            return list(self.checkpoint["class_names"])

        with self.label_map_path.open("r", encoding="utf-8") as f:
            label_map = json.load(f)

        return [
            label
            for _, label in sorted(
                label_map.items(),
                key=lambda item: int(item[0]),
            )
        ]

    def _load_model(self) -> MultimodalDeepClassifier:
        n_classes = int(self.checkpoint.get("n_classes", len(self.class_names)))

        model = MultimodalDeepClassifier(
            n_classes=n_classes,
        ).to(self.device)

        model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        model.eval()

        return model

    @staticmethod
    def _validate_columns(df: pd.DataFrame) -> None:
        required = [
            "spectrometer",
            "lifetime",
            "scattering_image",
            "size",
            "time_asymmetry",
        ]

        missing = [
            col for col in required
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns for multimodal inference: {missing}"
            )

    def build_inputs(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        self._validate_columns(df)

        X_spec = stack_vector_column(df, "spectrometer").astype(np.float32)
        X_life = stack_vector_column(df, "lifetime").astype(np.float32)

        X_scat = np.stack(
            df["scattering_image"].apply(crop_pad_scattering).values
        ).astype(np.float32)

        X_scalar_df = df[["size", "time_asymmetry"]].copy()
        X_scalar_df["size"] = pd.to_numeric(
            X_scalar_df["size"],
            errors="coerce",
        )
        X_scalar_df["time_asymmetry"] = pd.to_numeric(
            X_scalar_df["time_asymmetry"],
            errors="coerce",
        )

        if X_scalar_df.isna().any().any():
            raise ValueError(
                f"Missing scalar values:\n{X_scalar_df.isna().sum()}"
            )

        X_scalar = X_scalar_df.to_numpy(dtype=np.float32)

        X_spec = self.spec_scaler.transform(X_spec)
        X_life = self.life_scaler.transform(X_life)
        X_scat = self.scat_scaler.transform(X_scat)
        X_scalar = self.scalar_scaler.transform(X_scalar)

        return {
            "spectrometer": torch.tensor(
                X_spec,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            "lifetime": torch.tensor(
                X_life,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            "scattering": torch.tensor(
                X_scat,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1),
            "scalar": torch.tensor(
                X_scalar,
                dtype=torch.float32,
                device=self.device,
            ),
        }

    @torch.no_grad()
    def predict_particles(
        self,
        df: pd.DataFrame,
        batch_size: int = 512,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        outputs: list[pd.DataFrame] = []

        for start in range(0, len(df), batch_size):
            end = start + batch_size

            batch_df = df.iloc[start:end].copy()
            batch = self.build_inputs(batch_df)

            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()

            pred_idx = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)

            result = batch_df.copy()
            result["predicted_class_index"] = pred_idx
            result["predicted_label"] = [
                self.class_names[i] for i in pred_idx
            ]
            result["prediction_confidence"] = confidence

            for i, class_name in enumerate(self.class_names):
                safe_name = (
                    class_name
                    .replace(" ", "_")
                    .replace(".", "")
                    .replace("-", "_")
                )
                result[f"prob_{safe_name}"] = probabilities[:, i]

            outputs.append(result)

        return pd.concat(outputs, ignore_index=True)


def predict_raw_dataframe(
    df: pd.DataFrame,
    model_path: Path | str = Path("models/trained/multimodal_species_v1.pt"),
) -> pd.DataFrame:
    """
    Convenience function for quick tests.
    """
    engine = MultimodalInferenceEngine(
        model_path=model_path,
    )
    return engine.predict_particles(df)