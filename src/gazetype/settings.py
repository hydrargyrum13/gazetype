from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from gazetype.calibration import CalibrationModel
from gazetype.models import KeyboardLayout, Sensitivity


@dataclass(slots=True)
class AppSettings:
    camera_index: int = 0
    screen_name: str = ""
    screen_geometry: str = ""
    layout: KeyboardLayout = KeyboardLayout.TURKISH_Q
    sensitivity: Sensitivity = Sensitivity.FAST
    calibration: CalibrationModel | None = None

    def __post_init__(self) -> None:
        # Qt stores StrEnum values as plain strings inside QVariant/QComboBox.
        # Normalize at the boundary so persistence always sees real enums.
        if not isinstance(self.layout, KeyboardLayout):
            self.layout = KeyboardLayout(str(self.layout))
        if not isinstance(self.sensitivity, Sensitivity):
            self.sensitivity = Sensitivity(str(self.sensitivity))

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["layout"] = self.layout.value
        data["sensitivity"] = self.sensitivity.value
        data["calibration"] = self.calibration.to_dict() if self.calibration else None
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppSettings":
        calibration_data = data.get("calibration")
        return cls(
            camera_index=int(data.get("camera_index", 0)),
            screen_name=str(data.get("screen_name", "")),
            screen_geometry=str(data.get("screen_geometry", "")),
            layout=KeyboardLayout(str(data.get("layout", KeyboardLayout.TURKISH_Q.value))),
            sensitivity=Sensitivity(str(data.get("sensitivity", Sensitivity.FAST.value))),
            calibration=(
                CalibrationModel.from_dict(calibration_data)
                if isinstance(calibration_data, dict)
                else None
            ),
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
