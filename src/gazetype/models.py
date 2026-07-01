from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class KeyboardLayout(StrEnum):
    TURKISH_Q = "tr_q"
    ENGLISH_QWERTY = "en_qwerty"


class Sensitivity(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    STABLE = "stable"


@dataclass(frozen=True, slots=True)
class SensitivityProfile:
    minimum_frames: int
    minimum_ms: int
    saccade_velocity: float
    landing_velocity: float


SENSITIVITY_PROFILES: dict[Sensitivity, SensitivityProfile] = {
    Sensitivity.FAST: SensitivityProfile(2, 50, 1.45, 0.82),
    Sensitivity.BALANCED: SensitivityProfile(3, 90, 1.25, 0.68),
    Sensitivity.STABLE: SensitivityProfile(4, 130, 1.05, 0.55),
}


@dataclass(frozen=True, slots=True)
class GazePoint:
    timestamp_ms: int
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class VisionFrame:
    timestamp_ms: int
    features: tuple[float, ...]
    blink_left: float
    blink_right: float
    fps: float
