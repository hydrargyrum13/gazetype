from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io


@dataclass(frozen=True, slots=True)
class MpiiGazeSample:
    participant: str
    day: str
    filename: str
    eye: str
    image_path: Path
    gaze_vector: tuple[float, float, float]
    head_pose: tuple[float, float, float]


def _field(record: np.ndarray, name: str) -> np.ndarray:
    return record[name][0, 0]


def _filename_values(values: np.ndarray) -> list[str]:
    names = []
    for value in values.reshape(-1):
        if isinstance(value, np.ndarray):
            names.append(str(value.reshape(-1)[0]))
        else:
            names.append(str(value))
    return names


def normalized_mat_paths(data_dir: Path) -> list[Path]:
    root = data_dir / "Data" / "Normalized"
    if not root.exists():
        raise FileNotFoundError(f"MPIIGaze normalized directory not found: {root}")
    return sorted(root.glob("p*/day*.mat"))


def iter_normalized_samples(data_dir: Path, limit: int | None = None) -> Iterable[MpiiGazeSample]:
    emitted = 0
    for mat_path in normalized_mat_paths(data_dir):
        participant = mat_path.parent.name
        day = mat_path.stem
        mat = scipy.io.loadmat(mat_path, squeeze_me=False)
        filenames = _filename_values(mat["filenames"])
        data = mat["data"]
        for eye in ("left", "right"):
            record = _field(data, eye)
            gaze = _field(record, "gaze")
            pose = _field(record, "pose")
            for index, filename in enumerate(filenames):
                yield MpiiGazeSample(
                    participant=participant,
                    day=day,
                    filename=filename,
                    eye=eye,
                    image_path=data_dir / "Data" / "Original" / participant / day / filename,
                    gaze_vector=tuple(float(value) for value in gaze[index]),
                    head_pose=tuple(float(value) for value in pose[index]),
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def gaze_vector_to_angles(gaze_vector: tuple[float, float, float]) -> tuple[float, float]:
    gaze = np.asarray(gaze_vector, dtype=np.float64)
    gaze /= max(float(np.linalg.norm(gaze)), 1e-9)
    yaw = float(np.arctan2(gaze[0], -gaze[2]))
    pitch = float(np.arcsin(np.clip(gaze[1], -1.0, 1.0)))
    return yaw, pitch


def angles_to_pseudo_target(
    yaw: float,
    pitch: float,
    yaw_range: tuple[float, float] = (-0.5, 0.5),
    pitch_range: tuple[float, float] = (-0.35, 0.35),
) -> tuple[float, float]:
    yaw_min, yaw_max = yaw_range
    pitch_min, pitch_max = pitch_range
    x = (yaw - yaw_min) / max(yaw_max - yaw_min, 1e-9)
    y = 1.0 - (pitch - pitch_min) / max(pitch_max - pitch_min, 1e-9)
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def summarize(data_dir: Path, limit: int | None = None) -> dict[str, object]:
    samples = list(iter_normalized_samples(data_dir, limit=limit))
    yaws = []
    pitches = []
    missing_images = 0
    for sample in samples:
        yaw, pitch = gaze_vector_to_angles(sample.gaze_vector)
        yaws.append(yaw)
        pitches.append(pitch)
        if not sample.image_path.exists():
            missing_images += 1
    return {
        "sample_count": len(samples),
        "participant_count": len({sample.participant for sample in samples}),
        "missing_image_count": missing_images,
        "yaw_range": (float(np.min(yaws)), float(np.max(yaws))) if yaws else (0.0, 0.0),
        "pitch_range": (float(np.min(pitches)), float(np.max(pitches))) if pitches else (0.0, 0.0),
    }
