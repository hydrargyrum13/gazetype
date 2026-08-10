from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def app_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Gazetype"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "gazetype"


def default_training_samples_path() -> Path:
    return app_data_dir() / "training_samples.jsonl"


def _screen_geometry(value: str) -> dict[str, int] | str:
    parts = value.split(",")
    if len(parts) != 4:
        return value
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError:
        return value
    return {"x": x, "y": y, "width": width, "height": height}


@dataclass(frozen=True, slots=True)
class TrainingSampleWriter:
    path: Path
    screen_geometry: str
    camera_index: int

    def write(
        self,
        target: tuple[float, float],
        features: Iterable[float],
        quality: dict[str, object] | None = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, object] = {
            "timestamp": time.time(),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "features": [float(value) for value in features],
            "screen_geometry": _screen_geometry(self.screen_geometry),
            "camera_index": int(self.camera_index),
        }
        if quality:
            row["quality"] = quality
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
