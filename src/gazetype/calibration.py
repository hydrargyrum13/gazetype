from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CALIBRATION_TARGETS: tuple[tuple[float, float], ...] = (
    (0.08, 0.08), (0.50, 0.08), (0.92, 0.08),
    (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
    (0.08, 0.92), (0.50, 0.92), (0.92, 0.92),
)


def _basis(features: np.ndarray) -> np.ndarray:
    gx, gy, head_x, head_y = features.T
    return np.column_stack((
        np.ones(len(features)), gx, gy, gx * gx, gx * gy, gy * gy, head_x, head_y
    ))


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    coefficients: tuple[tuple[float, ...], tuple[float, ...]]

    @classmethod
    def fit(
        cls,
        features: Iterable[Iterable[float]],
        targets: Iterable[Iterable[float]],
        ridge: float = 1e-4,
    ) -> "CalibrationModel":
        feature_array = np.asarray(tuple(features), dtype=np.float64)
        target_array = np.asarray(tuple(targets), dtype=np.float64)
        if feature_array.ndim != 2 or feature_array.shape[1] != 4:
            raise ValueError("Calibration requires four gaze features per sample")
        if len(feature_array) < 8 or target_array.shape != (len(feature_array), 2):
            raise ValueError("Calibration requires at least eight paired samples")
        design = _basis(feature_array)
        regularizer = ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target_array)
        return cls(tuple(tuple(float(value) for value in coefficients[:, axis]) for axis in range(2)))

    def predict(self, features: Iterable[float]) -> tuple[float, float]:
        row = np.asarray(tuple(features), dtype=np.float64)
        if row.shape != (4,):
            raise ValueError("Expected four gaze features")
        design = _basis(row.reshape(1, 4))[0]
        coefficients = np.asarray(self.coefficients, dtype=np.float64).T
        result = design @ coefficients
        return float(np.clip(result[0], 0.0, 1.0)), float(np.clip(result[1], 0.0, 1.0))

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {"coefficients": [list(axis) for axis in self.coefficients]}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CalibrationModel":
        values = data["coefficients"]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("Invalid calibration coefficients")
        return cls(tuple(tuple(float(value) for value in axis) for axis in values))  # type: ignore[arg-type]

