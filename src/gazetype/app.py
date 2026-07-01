from __future__ import annotations

import sys
from collections import deque
from math import sqrt

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from gazetype.blink import DeliberateBlinkDetector
from gazetype.calibration import CalibrationModel, calibration_targets
from gazetype.input_windows import WindowsInputSender
from gazetype.keyboards import KeyboardGeometry
from gazetype.landing import LandingDetector
from gazetype.models import GazePoint, KeyboardLayout, SENSITIVITY_PROFILES, Sensitivity, VisionFrame
from gazetype.settings import AppSettings, SettingsStore
from gazetype.ui import (
    CalibrationWindow,
    KeyboardOverlay,
    SettingsWindow,
    ToggleWindow,
    TrackingWindow,
    show_error,
)
from gazetype.vision import CameraWorker


def head_motion_speed(
    previous: tuple[int, tuple[float, ...]] | None,
    timestamp_ms: int,
    features: tuple[float, ...],
) -> float:
    if previous is None:
        return 0.0
    previous_ms, previous_features = previous
    elapsed = max((timestamp_ms - previous_ms) / 1000.0, 0.001)
    weights = (1.0, 1.0, 1.8, 3.0, 1.4, 1.4)
    delta = sum(
        ((features[index] - previous_features[index]) * weight) ** 2
        for index, weight in zip(range(4, 10), weights, strict=True)
    ) ** 0.5
    return delta / elapsed


def eye_ratio_gains(model: CalibrationModel) -> tuple[float, float]:
    """Balance screen gain using eye-ratio movement measured during calibration."""
    horizontal_spread = max((model.feature_scale[0] + model.feature_scale[2]) / 2.0, 1e-6)
    vertical_spread = max((model.feature_scale[1] + model.feature_scale[3]) / 2.0, 1e-6)
    horizontal_gain = 100.0 * sqrt(vertical_spread / horizontal_spread)
    vertical_gain = 100.0 * sqrt(horizontal_spread / vertical_spread)
    return (
        max(50.0, min(horizontal_gain, 200.0)),
        max(50.0, min(vertical_gain, 250.0)),
    )


class GazetypeController:
    def __init__(self, application: QApplication):
        self.application = application
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.settings_window = SettingsWindow(self.settings)
        self.calibration_window = CalibrationWindow()
        self.overlay = KeyboardOverlay()
        self.toggle = ToggleWindow()
        self.tracking_window = TrackingWindow()
        self.worker: CameraWorker | None = None
        self.input_sender = WindowsInputSender()
        self.landing = LandingDetector(SENSITIVITY_PROFILES[self.settings.sensitivity])
        self.blink = DeliberateBlinkDetector()
        self.keyboard_enabled = False
        self.face_present = False
        self.screen = None
        self.recent_gaze: deque[tuple[float, float]] = deque(maxlen=3)
        self.previous_head_sample: tuple[int, tuple[float, ...]] | None = None
        self.head_motion_until_ms = 0

        self.settings_window.start_requested.connect(self.begin_calibration)
        self.calibration_window.completed.connect(self.finish_calibration)
        self.calibration_window.cancelled.connect(self.cancel_calibration)
        self.toggle.toggled.connect(self.toggle_keyboard)
        self.tracking_window.tuning_changed.connect(self.update_tuning)
        self._refresh_screens()
        self._create_tray()
        self.settings_window.show()

    def _refresh_screens(self) -> None:
        items = []
        for index, screen in enumerate(self.application.screens()):
            rect = screen.geometry()
            label = f"Ekran {index + 1} — {screen.name()} ({rect.width()}×{rect.height()})"
            geometry = f"{rect.x()},{rect.y()},{rect.width()},{rect.height()}"
            items.append((label, geometry))
        self.settings_window.set_screens(items, self.settings.screen_name)

    def _create_tray(self) -> None:
        icon = self.application.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray = QSystemTrayIcon(icon, self.application)
        self.tray.setToolTip("Gazetype")
        menu = QMenu()
        settings_action = QAction("Ayarlar / Yeniden Kalibre Et", menu)
        settings_action.triggered.connect(self.show_settings)
        quit_action = QAction("Çıkış", menu)
        quit_action.triggered.connect(self.shutdown)
        menu.addAction(settings_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self.show_settings() if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        self.tray.show()

    def begin_calibration(self, values: dict[str, object]) -> None:
        screens = self.application.screens()
        screen_index = int(values["screen_index"])
        if screen_index >= len(screens):
            self.settings_window.set_status("Seçilen ekran artık bağlı değil.", True)
            self.settings_window.unlock()
            self._refresh_screens()
            return
        self.screen = screens[screen_index]
        self.settings = AppSettings(
            camera_index=int(values["camera_index"]),
            screen_name=str(values["screen_name"]),
            screen_geometry=str(values["screen_geometry"]),
            layout=KeyboardLayout(str(values["layout"])),
            sensitivity=Sensitivity(str(values["sensitivity"])),
            calibration_point_count=int(values["calibration_point_count"]),
            calibration_mode=str(values["calibration_mode"]),
            gaze_average_count=int(values["gaze_average_count"]),
            auto_gaze_gain=bool(values["auto_gaze_gain"]),
            quadrilateral_eye_mapping=bool(values["quadrilateral_eye_mapping"]),
            horizontal_gain_percent=int(values["horizontal_gain_percent"]),
            vertical_gain_percent=int(values["vertical_gain_percent"]),
            vertical_offset_percent=int(values["vertical_offset_percent"]),
            head_compensation_percent=int(values["head_compensation_percent"]),
            head_motion_threshold_percent=int(values["head_motion_threshold_percent"]),
        )
        self.recent_gaze = deque(maxlen=self.settings.gaze_average_count)
        self.tracking_window.configure_tuning(self.settings)
        self.previous_head_sample = None
        self.head_motion_until_ms = 0
        self._stop_worker()
        self.worker = CameraWorker(
            self.settings.camera_index, self.settings.quadrilateral_eye_mapping
        )
        self.worker.frame_ready.connect(self.on_vision_frame)
        self.worker.tracking_preview.connect(self.tracking_window.set_frame)
        self.worker.face_presence.connect(self.on_face_presence)
        self.worker.error.connect(self.on_camera_error)
        self.worker.start()
        self.tracking_window.show()
        self.settings_window.showMinimized()
        calibration_keyboard = None
        if self.settings.calibration_mode == "keyboard":
            calibration_keyboard = KeyboardGeometry(self.settings.layout)
            targets = tuple(
                (key.x + key.width / 2, key.y + key.height / 2)
                for key in calibration_keyboard.keys
            )
        else:
            targets = calibration_targets(self.settings.calibration_point_count)
        self.calibration_window.begin(self.screen, targets, calibration_keyboard)

    def finish_calibration(self, features, targets) -> None:
        try:
            self.settings.calibration = CalibrationModel.fit(features, targets)
            self.store.save(self.settings)
        except ValueError as exc:
            self.on_camera_error(str(exc))
            return
        self.landing = LandingDetector(SENSITIVITY_PROFILES[self.settings.sensitivity])
        self.overlay.configure(self.screen, self.settings.layout)
        self.toggle.place(self.screen)
        self.keyboard_enabled = False
        self.toggle.set_enabled(False)
        self.settings_window.unlock()
        self.tray.showMessage(
            "Gazetype hazır",
            "Hedef uygulamayı seçin, ardından köşe düğmesine bakıp bilinçli göz kırpın.",
            QSystemTrayIcon.MessageIcon.Information,
            5000,
        )

    def cancel_calibration(self) -> None:
        self._stop_worker()
        self.tracking_window.hide()
        self.settings_window.unlock()
        self.settings_window.set_status("Kalibrasyon iptal edildi.")
        self.settings_window.show()

    def on_vision_frame(self, frame: VisionFrame) -> None:
        if self.calibration_window.isVisible():
            self.calibration_window.add_sample(frame.timestamp_ms, frame.features)
            return
        model = self.settings.calibration
        if model is None or self.screen is None:
            return
        compensated_features = list(frame.features)
        compensation = self.settings.head_compensation_percent / 100.0
        for index in range(4, len(compensated_features)):
            baseline = model.feature_mean[index]
            compensated_features[index] = baseline + (
                compensated_features[index] - baseline
            ) * compensation
        x, y = model.predict(compensated_features)
        if self.settings.auto_gaze_gain:
            horizontal_gain, vertical_gain = eye_ratio_gains(model)
        else:
            horizontal_gain = float(self.settings.horizontal_gain_percent)
            vertical_gain = float(self.settings.vertical_gain_percent)
        x = 0.5 + (x - 0.5) * horizontal_gain / 100.0
        y = (
            0.5
            + (y - 0.5) * vertical_gain / 100.0
            + self.settings.vertical_offset_percent / 100.0
        )
        x = max(0.0, min(x, 1.0))
        y = max(0.0, min(y, 1.0))
        motion_speed = head_motion_speed(
            self.previous_head_sample, frame.timestamp_ms, frame.features
        )
        self.previous_head_sample = (frame.timestamp_ms, frame.features)
        motion_threshold = 0.9 * self.settings.head_motion_threshold_percent / 100.0
        if motion_speed > motion_threshold:
            self.head_motion_until_ms = frame.timestamp_ms + 160
        self.recent_gaze.append((x, y))
        x = sum(point[0] for point in self.recent_gaze) / len(self.recent_gaze)
        y = sum(point[1] for point in self.recent_gaze) / len(self.recent_gaze)
        toggle_armed = x >= 0.87 and y <= 0.15
        eyes_closed = frame.blink_left >= self.blink.close_threshold and frame.blink_right >= self.blink.close_threshold
        if eyes_closed:
            self.landing.cancel_candidate()
        if self.blink.update(frame.timestamp_ms, frame.blink_left, frame.blink_right, toggle_armed):
            self.toggle_keyboard()
            return
        if not self.keyboard_enabled or eyes_closed:
            return
        if frame.timestamp_ms < self.head_motion_until_ms:
            self.landing.cancel_candidate()
            self.overlay.set_gaze_state(None, 0.0, frame.fps, (x, y))
            return
        key = self.overlay.keyboard.hit_test(x, y)
        selected = self.landing.update(GazePoint(frame.timestamp_ms, x, y), key.id if key else None)
        self.overlay.set_gaze_state(
            self.landing.candidate_key,
            self.landing.candidate_progress,
            frame.fps,
            (x, y),
        )
        if selected:
            selected_key = self.overlay.keyboard.by_id(selected)
            if not self.input_sender.send(selected_key.id, selected_key.output):
                self.tray.showMessage("Gazetype", "Tuş aktif uygulamaya gönderilemedi.")

    def on_face_presence(self, present: bool) -> None:
        self.face_present = present
        self.calibration_window.set_face_present(present)
        self.overlay.face_present = present
        if not present:
            self.recent_gaze.clear()
            self.previous_head_sample = None
            self.landing.reset()
            self.blink.reset()
            self.overlay.set_gaze_state(None, 0.0, 0.0, None)

    def update_tuning(self, values: dict[str, int]) -> None:
        for key, value in values.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, int(value))
        if self.settings.calibration is not None:
            self.store.save(self.settings)

    def on_camera_error(self, message: str) -> None:
        self.calibration_window.hide()
        self.tracking_window.hide()
        self.settings_window.unlock()
        self.settings_window.set_status(message, True)
        self.settings_window.show()
        show_error(self.settings_window, "Kamera hatası", message)

    def toggle_keyboard(self) -> None:
        if self.settings.calibration is None or self.screen is None:
            return
        self.keyboard_enabled = not self.keyboard_enabled
        self.landing.reset()
        self.recent_gaze.clear()
        self.previous_head_sample = None
        self.head_motion_until_ms = 0
        self.blink.reset()
        self.toggle.set_enabled(self.keyboard_enabled)
        if self.keyboard_enabled:
            self.overlay.show_non_activating()
        else:
            self.overlay.hide()

    def show_settings(self) -> None:
        self.keyboard_enabled = False
        self.overlay.hide()
        self.toggle.hide()
        self.tracking_window.hide()
        self._stop_worker()
        self.settings_window.unlock()
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()
        self._refresh_screens()

    def _stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.deleteLater()
            self.worker = None

    def shutdown(self) -> None:
        self.settings_window.stop_camera_previews()
        self._stop_worker()
        self.tracking_window.close()
        self.tray.hide()
        self.application.quit()


def main() -> int:
    application = QApplication(sys.argv)
    application.setApplicationName("Gazetype")
    application.setOrganizationName("Gazetype")
    application.setQuitOnLastWindowClosed(False)
    controller = GazetypeController(application)
    application.aboutToQuit.connect(controller._stop_worker)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
