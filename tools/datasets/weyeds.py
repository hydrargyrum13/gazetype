from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FEATURE_COLUMNS = tuple(f"feature_{index}" for index in range(10))
MANIFEST_NAMES = (
    "gazetype_manifest.csv",
    "gazetype_manifest.jsonl",
    "manifest.csv",
    "manifest.jsonl",
    "samples.csv",
    "samples.jsonl",
)


@dataclass(frozen=True, slots=True)
class NormalizedGazeSample:
    features: tuple[float, ...]
    target: tuple[float, float]
    image_path: str = ""
    face_path: str = ""
    left_eye_path: str = ""
    right_eye_path: str = ""
    screen_width: float | None = None
    screen_height: float | None = None


def find_manifest(data_dir: Path) -> Path:
    for name in MANIFEST_NAMES:
        candidate = data_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "WEyeDS normalized manifest not found. Expected one of: "
        + ", ".join(MANIFEST_NAMES)
    )


def _features_from_row(row: dict[str, object]) -> tuple[float, ...]:
    raw_features = row.get("features")
    if isinstance(raw_features, str) and raw_features:
        parsed = json.loads(raw_features)
        if isinstance(parsed, list) and len(parsed) == 10:
            return tuple(float(value) for value in parsed)
    if isinstance(raw_features, list) and len(raw_features) == 10:
        return tuple(float(value) for value in raw_features)
    if all(column in row for column in FEATURE_COLUMNS):
        return tuple(float(row[column]) for column in FEATURE_COLUMNS)
    raise ValueError(
        "Normalized WEyeDS rows must include a 10-value 'features' field or "
        "feature_0..feature_9 columns. Raw image feature extraction is not part "
        "of this baseline trainer yet."
    )


def _optional_float(row: dict[str, object], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def _sample_from_row(row: dict[str, object]) -> NormalizedGazeSample:
    if "target_x" not in row or "target_y" not in row:
        raise ValueError("Normalized WEyeDS rows must include target_x and target_y")
    return NormalizedGazeSample(
        features=_features_from_row(row),
        target=(float(row["target_x"]), float(row["target_y"])),
        image_path=str(row.get("image_path", "") or ""),
        face_path=str(row.get("face_path", "") or ""),
        left_eye_path=str(row.get("left_eye_path", "") or ""),
        right_eye_path=str(row.get("right_eye_path", "") or ""),
        screen_width=_optional_float(row, "screen_width"),
        screen_height=_optional_float(row, "screen_height"),
    )


def load_normalized_manifest(path: Path) -> list[NormalizedGazeSample]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [_sample_from_row(dict(row)) for row in csv.DictReader(handle)]
    if path.suffix.lower() == ".jsonl":
        samples = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_number} is not an object")
                samples.append(_sample_from_row(row))
        return samples
    raise ValueError("WEyeDS normalized manifest must be .csv or .jsonl")


def load_dataset(data_dir: Path) -> list[NormalizedGazeSample]:
    return load_normalized_manifest(find_manifest(data_dir))


def arrays(samples: Iterable[NormalizedGazeSample]) -> tuple[list[tuple[float, ...]], list[tuple[float, float]]]:
    sample_list = list(samples)
    return [sample.features for sample in sample_list], [sample.target for sample in sample_list]
