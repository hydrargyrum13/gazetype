from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from gazetype.calibration import (
    DEFAULT_CALIBRATION_POINT_COUNT,
    MAXIMUM_CALIBRATION_POINTS,
    MINIMUM_CALIBRATION_POINTS,
    CalibrationModel,
)
from gazetype.models import KeyboardLayout, Sensitivity


@dataclass(slots=True)
class AppSettings:
    camera_index: int = 0
    screen_name: str = ""
    screen_geometry: str = ""
    layout: KeyboardLayout = KeyboardLayout.TURKISH_Q
    sensitivity: Sensitivity = Sensitivity.FAST
    calibration_point_count: int = DEFAULT_CALIBRATION_POINT_COUNT
    calibration_mode: str = "grid"
    gaze_average_count: int = 3
    auto_gaze_gain: bool = True
    quadrilateral_eye_mapping: bool = True
    binocular_stabilization: bool = True
    adaptive_gaze_filter: bool = True
    robust_calibration: bool = True
    horizontal_gain_percent: int = 100
    vertical_gain_percent: int = 130
    vertical_offset_percent: int = 0
    head_compensation_percent: int = 100
    head_motion_threshold_percent: int = 100
    calibration: CalibrationModel | None = None

    def __post_init__(self) -> None:
        # Qt stores StrEnum values as plain strings inside QVariant/QComboBox.
        # Normalize at the boundary so persistence always sees real enums.
        if not isinstance(self.layout, KeyboardLayout):
            self.layout = KeyboardLayout(str(self.layout))
        if not isinstance(self.sensitivity, Sensitivity):
            self.sensitivity = Sensitivity(str(self.sensitivity))
        self.calibration_point_count = max(
            MINIMUM_CALIBRATION_POINTS,
            min(int(self.calibration_point_count), MAXIMUM_CALIBRATION_POINTS),
        )
        self.gaze_average_count = max(1, min(int(self.gaze_average_count), 30))
        self.horizontal_gain_percent = max(50, min(int(self.horizontal_gain_percent), 200))
        self.vertical_gain_percent = max(50, min(int(self.vertical_gain_percent), 250))
        self.vertical_offset_percent = max(-25, min(int(self.vertical_offset_percent), 25))
        self.head_compensation_percent = max(0, min(int(self.head_compensation_percent), 150))
        self.head_motion_threshold_percent = max(
            40, min(int(self.head_motion_threshold_percent), 200)
        )
        if self.calibration_mode not in {"grid", "keyboard"}:
            self.calibration_mode = "grid"

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["layout"] = self.layout.value
        data["sensitivity"] = self.sensitivity.value
        data["calibration"] = self.calibration.to_dict() if self.calibration else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppSettings":
        calibration_data = data.get("calibration")
        calibration = None
        if isinstance(calibration_data, dict):
            try:
                calibration = CalibrationModel.from_dict(calibration_data)
            except (KeyError, TypeError, ValueError):
                calibration = None
        return cls(
            camera_index=int(data.get("camera_index", 0)),
            screen_name=str(data.get("screen_name", "")),
            screen_geometry=str(data.get("screen_geometry", "")),
            layout=KeyboardLayout(str(data.get("layout", KeyboardLayout.TURKISH_Q.value))),
            sensitivity=Sensitivity(str(data.get("sensitivity", Sensitivity.FAST.value))),
            calibration_point_count=int(
                data.get("calibration_point_count", DEFAULT_CALIBRATION_POINT_COUNT)
            ),
            calibration_mode=str(data.get("calibration_mode", "grid")),
            gaze_average_count=int(data.get("gaze_average_count", 3)),
            auto_gaze_gain=bool(data.get("auto_gaze_gain", True)),
            quadrilateral_eye_mapping=bool(data.get("quadrilateral_eye_mapping", True)),
            binocular_stabilization=bool(data.get("binocular_stabilization", True)),
            adaptive_gaze_filter=bool(data.get("adaptive_gaze_filter", True)),
            robust_calibration=bool(data.get("robust_calibration", True)),
            horizontal_gain_percent=int(data.get("horizontal_gain_percent", 100)),
            vertical_gain_percent=int(data.get("vertical_gain_percent", 130)),
            vertical_offset_percent=int(data.get("vertical_offset_percent", 0)),
            head_compensation_percent=int(data.get("head_compensation_percent", 100)),
            head_motion_threshold_percent=int(
                data.get("head_motion_threshold_percent", 100)
            ),
            calibration=calibration,
        )


class SettingsStore:
    def __init__(self, path: Path | None = None):
        if path is None:
            base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
            path = base / "Gazetype" / "settings.json"
        self.path = path

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            return AppSettings.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)
