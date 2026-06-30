from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CALIBRATION_MODEL_VERSION = 2
DEFAULT_CALIBRATION_POINT_COUNT = 25
MINIMUM_CALIBRATION_POINTS = 20
MAXIMUM_CALIBRATION_POINTS = 49


def calibration_targets(count: int) -> tuple[tuple[float, float], ...]:
    if not MINIMUM_CALIBRATION_POINTS <= count <= MAXIMUM_CALIBRATION_POINTS:
        raise ValueError("Calibration point count must be between 20 and 49")
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    xs = np.linspace(0.06, 0.94, columns)
    ys = np.linspace(0.06, 0.94, rows)
    targets: list[tuple[float, float]] = []
    for row_index, y in enumerate(ys):
        row_xs = xs if row_index % 2 == 0 else xs[::-1]
        targets.extend((float(x), float(y)) for x in row_xs)
    return tuple(targets[:count])


CALIBRATION_TARGETS = calibration_targets(DEFAULT_CALIBRATION_POINT_COUNT)
BASIS_SIZE = 12


def _basis(features: np.ndarray) -> np.ndarray:
    gx, gy, head_x, head_y = features.T
    return np.column_stack((
        np.ones(len(features)),
        gx,
        gy,
        gx * gx,
        gx * gy,
        gy * gy,
        gx * gx * gx,
        gx * gx * gy,
        gx * gy * gy,
        gy * gy * gy,
        head_x,
        head_y,
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
        if len(feature_array) < MINIMUM_CALIBRATION_POINTS or target_array.shape != (len(feature_array), 2):
            raise ValueError(
                f"Calibration requires at least {MINIMUM_CALIBRATION_POINTS} paired samples"
            )
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

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": CALIBRATION_MODEL_VERSION,
            "coefficients": [list(axis) for axis in self.coefficients],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CalibrationModel":
        if data.get("model_version") != CALIBRATION_MODEL_VERSION:
            raise ValueError("Calibration model is outdated")
        values = data["coefficients"]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(not isinstance(axis, list) or len(axis) != BASIS_SIZE for axis in values)
        ):
            raise ValueError("Invalid calibration coefficients")
        return cls(tuple(tuple(float(value) for value in axis) for axis in values))  # type: ignore[arg-type]
