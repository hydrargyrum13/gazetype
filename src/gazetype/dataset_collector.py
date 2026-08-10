from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from gazetype.models import VisionFrame
from gazetype.training_data import TrainingSampleWriter, default_training_samples_path
from gazetype.vision import CameraWorker


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_rect(cls, rect: QRect) -> "ScreenGeometry":
        return cls(rect.x(), rect.y(), rect.width(), rect.height())

    def as_settings_value(self) -> str:
        return f"{self.x},{self.y},{self.width},{self.height}"


def normalized_target_for_position(
    position: QPoint,
    geometry: ScreenGeometry,
) -> tuple[float, float]:
    x = (position.x() - geometry.x) / max(geometry.width, 1)
    y = (position.y() - geometry.y) / max(geometry.height, 1)
    return max(0.0, min(float(x), 1.0)), max(0.0, min(float(y), 1.0))


class MouseDatasetCollector(QWidget):
    def __init__(
        self,
        camera_index: int,
        output_path: Path,
        screen_geometry: ScreenGeometry,
        quadrilateral_eye_mapping: bool,
    ):
        super().__init__()
        self.camera_index = camera_index
        self.screen_geometry = screen_geometry
        self.writer = TrainingSampleWriter(
            output_path,
            screen_geometry.as_settings_value(),
            camera_index,
        )
        self.latest_frame: VisionFrame | None = None
        self.face_present = False
        self.sample_count = 0
        self.last_target: tuple[float, float] | None = None
        self.status = "Kamera baslatiliyor"
        self.worker = CameraWorker(camera_index, quadrilateral_eye_mapping)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.face_presence.connect(self.on_face_presence)
        self.worker.error.connect(self.on_camera_error)
        self.worker.start()

        self.setWindowTitle("Gazetype dataset collector")
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setGeometry(
            screen_geometry.x,
            screen_geometry.y,
            screen_geometry.width,
            screen_geometry.height,
        )

    def on_frame(self, frame: VisionFrame) -> None:
        self.latest_frame = frame
        if self.face_present:
            self.status = f"Hazir - FPS {frame.fps:.1f}"
        self.update()

    def on_face_presence(self, present: bool) -> None:
        self.face_present = present
        self.status = "Hazir" if present else "Yuz bulunamadi"
        self.update()

    def on_camera_error(self, message: str) -> None:
        self.status = message
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        frame = self.latest_frame
        if frame is None or not self.face_present:
            self.status = "Kayit yok: yuz/feature bekleniyor"
            self.update()
            return
        global_position = event.globalPosition().toPoint()
        target = normalized_target_for_position(global_position, self.screen_geometry)
        self.writer.write(
            target,
            frame.features,
            {
                "source": "mouse_dataset_collector",
                "fps": frame.fps,
                "frame_timestamp_ms": frame.timestamp_ms,
                "blink_left": frame.blink_left,
                "blink_right": frame.blink_right,
            },
        )
        self.sample_count += 1
        self.last_target = target
        self.status = f"Kaydedildi: {self.sample_count}"
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.worker.deleteLater()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(3, 6, 12, 45))
        painter.setPen(QColor(44, 201, 151, 230))
        painter.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        painter.drawText(
            24,
            36,
            f"{self.status} | Ornek: {self.sample_count} | Sol tik kaydet | Esc cikis",
        )
        if self.last_target is not None:
            x = int(self.last_target[0] * self.width())
            y = int(self.last_target[1] * self.height())
            painter.setPen(QColor(255, 255, 255, 230))
            painter.drawEllipse(QPoint(x, y), 12, 12)
            painter.drawLine(x - 20, y, x + 20, y)
            painter.drawLine(x, y - 20, x, y + 20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect personal Gazetype screen-target samples by clicking where you look."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--screen-index", type=int, default=0)
    parser.add_argument("--out", type=Path, default=default_training_samples_path())
    parser.add_argument(
        "--classic-eye-ratio",
        action="store_true",
        help="Use the older rectangular eye-ratio feature extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    application = QApplication(sys.argv)
    screens = application.screens()
    if not screens:
        raise SystemExit("Ekran bulunamadi.")
    screen_index = max(0, min(args.screen_index, len(screens) - 1))
    geometry = ScreenGeometry.from_rect(screens[screen_index].geometry())
    collector = MouseDatasetCollector(
        args.camera_index,
        args.out,
        geometry,
        not args.classic_eye_ratio,
    )
    collector.showFullScreen()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
