from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QTimer
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


def moving_target_at_time(elapsed_seconds: float) -> tuple[float, float]:
    x = 0.5 + 0.43 * np.sin(elapsed_seconds * 0.73)
    y = 0.5 + 0.39 * np.sin(elapsed_seconds * 1.07 + 0.8)
    return float(np.clip(x, 0.04, 0.96)), float(np.clip(y, 0.05, 0.95))


class MouseDatasetCollector(QWidget):
    def __init__(
        self,
        camera_index: int,
        output_path: Path,
        screen_geometry: ScreenGeometry,
        quadrilateral_eye_mapping: bool,
        samples_per_capture: int,
        guided_targets: tuple[tuple[float, float], ...] = (),
        moving_target: bool = False,
        duration_seconds: int = 90,
        reaction_lag_ms: int = 250,
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
        self.frame_buffer: deque[VisionFrame] = deque(maxlen=max(samples_per_capture * 2, 12))
        self.face_present = False
        self.sample_count = 0
        self.capture_count = 0
        self.last_target: tuple[float, float] | None = None
        self.samples_per_capture = max(1, samples_per_capture)
        self.guided_targets = guided_targets
        self.guided_index = 0
        self.moving_target = moving_target
        self.duration_seconds = max(10, duration_seconds)
        self.reaction_lag_ms = max(0, reaction_lag_ms)
        self.started_at = time.perf_counter()
        self.target_history: deque[tuple[float, tuple[float, float]]] = deque(maxlen=300)
        self.current_target: tuple[float, float] | None = None
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
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setGeometry(
            screen_geometry.x,
            screen_geometry.y,
            screen_geometry.width,
            screen_geometry.height,
        )
        self.target_timer = QTimer(self)
        self.target_timer.timeout.connect(self.update_target)
        self.target_timer.start(33)

    def update_target(self) -> None:
        if not self.moving_target:
            self.update()
            return
        elapsed = time.perf_counter() - self.started_at
        if elapsed >= self.duration_seconds:
            self.status = f"Bitti: {self.sample_count} ornek"
            self.update()
            self.close()
            return
        self.current_target = moving_target_at_time(elapsed)
        self.target_history.append((time.perf_counter(), self.current_target))
        self.status = f"Takip et - {self.sample_count} frame"
        self.update()

    def on_frame(self, frame: VisionFrame) -> None:
        self.latest_frame = frame
        if self.face_present:
            self.frame_buffer.append(frame)
            if self.moving_target:
                self.capture_lagged_moving_frame(frame)
        if self.face_present and not self.moving_target:
            self.status = f"Hazir - FPS {frame.fps:.1f}"
        self.update()

    def lagged_target(self) -> tuple[float, float] | None:
        if not self.target_history:
            return self.current_target
        cutoff = time.perf_counter() - self.reaction_lag_ms / 1000.0
        candidate = self.target_history[0][1]
        for timestamp, target in self.target_history:
            if timestamp > cutoff:
                break
            candidate = target
        return candidate

    def capture_lagged_moving_frame(self, frame: VisionFrame) -> None:
        target = self.lagged_target()
        if target is None:
            return
        self.writer.write(
            target,
            frame.features,
            {
                "source": "moving_target_dataset_collector",
                "fps": frame.fps,
                "frame_timestamp_ms": frame.timestamp_ms,
                "blink_left": frame.blink_left,
                "blink_right": frame.blink_right,
                "reaction_lag_ms": self.reaction_lag_ms,
            },
        )
        self.sample_count += 1
        self.last_target = target

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
        if self.guided_targets:
            self.capture_current_guided_target()
            return
        self.capture_target(normalized_target_for_position(
            event.globalPosition().toPoint(),
            self.screen_geometry,
        ))

    def capture_current_guided_target(self) -> None:
        if self.guided_index >= len(self.guided_targets):
            self.close()
            return
        if self.capture_target(self.guided_targets[self.guided_index]):
            self.guided_index += 1
            if self.guided_index >= len(self.guided_targets):
                self.status = f"Bitti: {self.sample_count} ornek"
                self.update()

    def capture_target(self, target: tuple[float, float]) -> bool:
        frame = self.latest_frame
        if frame is None or not self.face_present or not self.frame_buffer:
            self.status = "Kayit yok: yuz/feature bekleniyor"
            self.update()
            return False
        selected = tuple(self.frame_buffer)[-self.samples_per_capture:]
        for buffered_frame in selected:
            self.writer.write(
                target,
                buffered_frame.features,
                {
                    "source": "mouse_dataset_collector",
                    "fps": buffered_frame.fps,
                    "frame_timestamp_ms": buffered_frame.timestamp_ms,
                    "blink_left": buffered_frame.blink_left,
                    "blink_right": buffered_frame.blink_right,
                    "capture_sample_count": len(selected),
                    "guided": bool(self.guided_targets),
                },
            )
        self.sample_count += len(selected)
        self.capture_count += 1
        self.last_target = target
        self.frame_buffer.clear()
        self.status = f"Kaydedildi: {self.capture_count} hedef / {self.sample_count} ornek"
        self.update()
        return True

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_Space and self.guided_targets:
            self.capture_current_guided_target()
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
            f"{self.status} | Esc cikis",
        )
        if self.moving_target and self.current_target is not None:
            x = int(self.current_target[0] * self.width())
            y = int(self.current_target[1] * self.height())
            painter.setPen(QColor(236, 255, 249, 245))
            painter.setBrush(QColor(44, 201, 151, 230))
            painter.drawEllipse(QPoint(x, y), 18, 18)
            painter.setBrush(QColor(255, 255, 255, 245))
            painter.drawEllipse(QPoint(x, y), 5, 5)
            return
        if self.guided_targets and self.guided_index < len(self.guided_targets):
            target = self.guided_targets[self.guided_index]
            x = int(target[0] * self.width())
            y = int(target[1] * self.height())
            painter.setPen(QColor(236, 255, 249, 245))
            painter.setBrush(QColor(44, 201, 151, 220))
            painter.drawEllipse(QPoint(x, y), 18, 18)
            painter.setBrush(QColor(255, 255, 255, 245))
            painter.drawEllipse(QPoint(x, y), 5, 5)
            painter.drawText(24, 66, f"Hedef {self.guided_index + 1}/{len(self.guided_targets)}")
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
    parser.add_argument("--samples-per-capture", type=int, default=6)
    parser.add_argument("--guided", action="store_true", help="Show balanced screen targets instead of using click position.")
    parser.add_argument("--moving", action="store_true", help="Record every frame while a delayed moving target crosses the screen.")
    parser.add_argument("--target-count", type=int, default=120)
    parser.add_argument("--duration-seconds", type=int, default=90)
    parser.add_argument("--reaction-lag-ms", type=int, default=250)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--classic-eye-ratio",
        action="store_true",
        help="Use the older rectangular eye-ratio feature extraction.",
    )
    return parser.parse_args()


def guided_target_sequence(count: int, seed: int) -> tuple[tuple[float, float], ...]:
    columns = max(5, int(round(count ** 0.5)))
    rows = max(4, int(np.ceil(count / columns)))
    xs = np.linspace(0.05, 0.95, columns)
    ys = np.linspace(0.06, 0.94, rows)
    targets = [(float(x), float(y)) for y in ys for x in xs]
    rng = random.Random(seed)
    rng.shuffle(targets)
    return tuple(targets[:count])


def main() -> int:
    args = parse_args()
    application = QApplication(sys.argv)
    screens = application.screens()
    if not screens:
        raise SystemExit("Ekran bulunamadi.")
    screen_index = max(0, min(args.screen_index, len(screens) - 1))
    geometry = ScreenGeometry.from_rect(screens[screen_index].geometry())
    guided_targets = guided_target_sequence(args.target_count, args.seed) if args.guided else ()
    collector = MouseDatasetCollector(
        args.camera_index,
        args.out,
        geometry,
        not args.classic_eye_ratio,
        args.samples_per_capture,
        guided_targets,
        args.moving,
        args.duration_seconds,
        args.reaction_lag_ms,
    )
    collector.showFullScreen()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
