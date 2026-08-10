from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from gazetype.calibration import CalibrationAdapterModel, CalibrationModel, GAZE_FEATURE_COUNT


class GazePredictor(Protocol):
    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        ...


def _model_design(features: np.ndarray, polynomial_degree: int) -> np.ndarray:
    columns = [np.ones(len(features))]
    columns.extend(features[:, index] for index in range(features.shape[1]))
    if polynomial_degree >= 2:
        for first in range(features.shape[1]):
            for second in range(first, features.shape[1]):
                columns.append(features[:, first] * features[:, second])
    return np.column_stack(columns)


@dataclass(frozen=True, slots=True)
class CalibrationPredictor:
    calibration: CalibrationModel

    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        return self.calibration.predict(features)


@dataclass(frozen=True, slots=True)
class NpzGazePredictor:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    weights: tuple[tuple[float, float], ...]
    polynomial_degree: int = 1

    @classmethod
    def load(cls, path: Path) -> "NpzGazePredictor":
        with np.load(path) as data:
            feature_mean = np.asarray(data["mean"], dtype=np.float64)
            feature_scale = np.asarray(data["std"], dtype=np.float64)
            weights = np.asarray(data["weights"], dtype=np.float64)
            polynomial_degree = int(data["polynomial_degree"]) if "polynomial_degree" in data else 1
        expected_columns = 1 + GAZE_FEATURE_COUNT
        if polynomial_degree >= 2:
            expected_columns += GAZE_FEATURE_COUNT * (GAZE_FEATURE_COUNT + 1) // 2
        if (
            feature_mean.shape != (GAZE_FEATURE_COUNT,)
            or feature_scale.shape != (GAZE_FEATURE_COUNT,)
            or weights.shape != (expected_columns, 2)
        ):
            raise ValueError("Invalid Gazetype general gaze model")
        return cls(
            tuple(float(value) for value in feature_mean),
            tuple(float(value) for value in np.maximum(feature_scale, 1e-6)),
            tuple((float(row[0]), float(row[1])) for row in weights),
            polynomial_degree,
        )

    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        row = np.asarray(tuple(features), dtype=np.float64)
        if row.shape != (GAZE_FEATURE_COUNT,):
            raise ValueError(f"Expected {GAZE_FEATURE_COUNT} gaze features")
        normalized = np.clip(
            (row - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale), -5.0, 5.0
        )
        design = _model_design(normalized.reshape(1, GAZE_FEATURE_COUNT), self.polynomial_degree)[0]
        result = design @ np.asarray(self.weights, dtype=np.float64)
        return float(np.clip(result[0], 0.0, 1.0)), float(np.clip(result[1], 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class AdapterPredictor:
    general: GazePredictor
    adapter: CalibrationAdapterModel
    fallback: CalibrationModel | None = None

    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        try:
            base = self.general.predict(features)
            return self.adapter.predict(base, features)
        except ValueError:
            if self.fallback is not None:
                return self.fallback.predict(features)
            raise


def configured_general_model_path(path: str = "") -> Path | None:
    configured = path or os.environ.get("GAZETYPE_GENERAL_MODEL", "")
    if not configured:
        return None
    return Path(configured).expanduser()


def load_general_predictor(path: Path | None) -> NpzGazePredictor | None:
    if path is None or not path.exists():
        return None
    return NpzGazePredictor.load(path)


def build_runtime_predictor(
    calibration: CalibrationModel,
    use_general_model: bool,
    model_path: str,
    adapter: CalibrationAdapterModel | None,
) -> GazePredictor:
    if use_general_model and adapter is not None:
        general = load_general_predictor(configured_general_model_path(model_path))
        if general is not None:
            return AdapterPredictor(general, adapter, fallback=calibration)
    return CalibrationPredictor(calibration)
