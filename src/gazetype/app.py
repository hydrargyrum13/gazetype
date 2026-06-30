from __future__ import annotations

import sys

from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from gazetype.blink import DeliberateBlinkDetector
from gazetype.calibration import CalibrationModel
from gazetype.input_windows import WindowsInputSender
from gazetype.landing import LandingDetector
from gazetype.models import GazePoint, SENSITIVITY_PROFILES, VisionFrame
from gazetype.settings import AppSettings, SettingsStore
from gazetype.ui import CalibrationWindow, KeyboardOverlay, SettingsWindow, ToggleWindow, show_error
from gazetype.vision import CameraWorker


class GazetypeController:
    def __init__(self, application: QApplication):
        self.application = application
        self.store = SettingsStore()
        self.settings = self.store.load()
        self.settings_window = SettingsWindow(self.settings)
        self.calibration_window = CalibrationWindow()
        self.overlay = KeyboardOverlay()
        self.toggle = ToggleWindow()
        self.worker: CameraWorker | None = None
        self.input_sender = WindowsInputSender()
        self.landing = LandingDetector(SENSITIVITY_PROFILES[self.settings.sensitivity])
        self.blink = DeliberateBlinkDetector()
        self.keyboard_enabled = False
        self.face_present = False
        self.screen = None

        self.settings_window.start_requested.connect(self.begin_calibration)
        self.calibration_window.completed.connect(self.finish_calibration)
        self.calibration_window.cancelled.connect(self.cancel_calibration)
        self.toggle.toggled.connect(self.toggle_keyboard)
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
            layout=values["layout"],
            sensitivity=values["sensitivity"],
        )
        self._stop_worker()
        self.worker = CameraWorker(self.settings.camera_index)
        self.worker.frame_ready.connect(self.on_vision_frame)
        self.worker.face_presence.connect(self.on_face_presence)
        self.worker.error.connect(self.on_camera_error)
        self.worker.start()
        self.settings_window.hide()
        self.calibration_window.begin(self.screen)

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
        x, y = model.predict(frame.features)
        toggle_armed = x >= 0.87 and y <= 0.15
        eyes_closed = frame.blink_left >= self.blink.close_threshold and frame.blink_right >= self.blink.close_threshold
        if eyes_closed:
            self.landing.cancel_candidate()
        if self.blink.update(frame.timestamp_ms, frame.blink_left, frame.blink_right, toggle_armed):
            self.toggle_keyboard()
            return
        if not self.keyboard_enabled or eyes_closed:
            return
        key = self.overlay.keyboard.hit_test(x, y)
        selected = self.landing.update(GazePoint(frame.timestamp_ms, x, y), key.id if key else None)
        self.overlay.set_gaze_state(
            self.landing.candidate_key,
            self.landing.candidate_progress,
            frame.fps,
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
            self.landing.reset()
            self.blink.reset()
            self.overlay.set_gaze_state(None, 0.0, 0.0)

    def on_camera_error(self, message: str) -> None:
        self.calibration_window.hide()
        self.settings_window.unlock()
        self.settings_window.set_status(message, True)
        self.settings_window.show()
        show_error(self.settings_window, "Kamera hatası", message)

    def toggle_keyboard(self) -> None:
        if self.settings.calibration is None or self.screen is None:
            return
        self.keyboard_enabled = not self.keyboard_enabled
        self.landing.reset()
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
        self._stop_worker()
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

