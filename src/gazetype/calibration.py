from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CALIBRATION_MODEL_VERSION = 4
GAZE_FEATURE_COUNT = 10
MINIMUM_FEATURE_SCALES = np.asarray(
    (0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.015, 0.01, 0.015, 0.015)
)
DEFAULT_CALIBRATION_POINT_COUNT = 25
MINIMUM_CALIBRATION_POINTS = 20
MAXIMUM_CALIBRATION_POINTS = 81


def calibration_targets(count: int) -> tuple[tuple[float, float], ...]:
    if not MINIMUM_CALIBRATION_POINTS <= count <= MAXIMUM_CALIBRATION_POINTS:
        raise ValueError(f"Calibration point count must be between 20 and {MAXIMUM_CALIBRATION_POINTS}")
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
BASIS_SIZE = 36


def _basis(features: np.ndarray) -> np.ndarray:
    left_x, left_y, right_x, right_y, head_x, head_y, roll, scale, yaw, pitch = features.T
    gx = (left_x + right_x) / 2
    gy = (left_y + right_y) / 2
    disparity_x = left_x - right_x
    disparity_y = left_y - right_y
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
        disparity_x,
        disparity_y,
        head_x,
        head_y,
        roll,
        scale,
        yaw,
        pitch,
        gx * head_x,
        gx * head_y,
        gx * roll,
        gx * scale,
        gy * head_x,
        gy * head_y,
        gy * roll,
        gy * scale,
        gx * yaw,
        gx * pitch,
        gy * yaw,
        gy * pitch,
        head_x * head_x,
        head_y * head_y,
        roll * roll,
        scale * scale,
        yaw * yaw,
        pitch * pitch,
    ))


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    coefficients: tuple[tuple[float, ...], tuple[float, ...]]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        features: Iterable[Iterable[float]],
        targets: Iterable[Iterable[float]],
        ridge: float = 1e-2,
    ) -> "CalibrationModel":
        feature_array = np.asarray(tuple(features), dtype=np.float64)
        target_array = np.asarray(tuple(targets), dtype=np.float64)
        if feature_array.ndim != 2 or feature_array.shape[1] != GAZE_FEATURE_COUNT:
            raise ValueError(f"Calibration requires {GAZE_FEATURE_COUNT} gaze features per sample")
        if len(feature_array) < MINIMUM_CALIBRATION_POINTS or target_array.shape != (len(feature_array), 2):
            raise ValueError(
                f"Calibration requires at least {MINIMUM_CALIBRATION_POINTS} paired samples"
            )
        feature_mean = np.mean(feature_array, axis=0)
        feature_scale = np.maximum(np.std(feature_array, axis=0), MINIMUM_FEATURE_SCALES)
        design = _basis((feature_array - feature_mean) / feature_scale)
        regularizer = ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target_array)
        return cls(
            tuple(tuple(float(value) for value in coefficients[:, axis]) for axis in range(2)),
            tuple(float(value) for value in feature_mean),
            tuple(float(value) for value in feature_scale),
        )

    def predict(self, features: Iterable[float]) -> tuple[float, float]:
        row = np.asarray(tuple(features), dtype=np.float64)
        if row.shape != (GAZE_FEATURE_COUNT,):
            raise ValueError(f"Expected {GAZE_FEATURE_COUNT} gaze features")
        normalized = np.clip(
            (row - np.asarray(self.feature_mean)) / np.asarray(self.feature_scale), -4.0, 4.0
        )
        design = _basis(normalized.reshape(1, GAZE_FEATURE_COUNT))[0]
        coefficients = np.asarray(self.coefficients, dtype=np.float64).T
        result = design @ coefficients
        return float(np.clip(result[0], 0.0, 1.0)), float(np.clip(result[1], 0.0, 1.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_version": CALIBRATION_MODEL_VERSION,
            "coefficients": [list(axis) for axis in self.coefficients],
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CalibrationModel":
        if data.get("model_version") != CALIBRATION_MODEL_VERSION:
            raise ValueError("Calibration model is outdated")
        values = data["coefficients"]
        feature_mean = data.get("feature_mean")
        feature_scale = data.get("feature_scale")
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(not isinstance(axis, list) or len(axis) != BASIS_SIZE for axis in values)
            or not isinstance(feature_mean, list)
            or len(feature_mean) != GAZE_FEATURE_COUNT
            or not isinstance(feature_scale, list)
            or len(feature_scale) != GAZE_FEATURE_COUNT
        ):
            raise ValueError("Invalid calibration coefficients")
        return cls(
            tuple(tuple(float(value) for value in axis) for axis in values),  # type: ignore[arg-type]
            tuple(float(value) for value in feature_mean),
            tuple(float(value) for value in feature_scale),
        )
