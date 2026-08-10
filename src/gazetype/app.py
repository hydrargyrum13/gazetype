from __future__ import annotations

import sys
from collections import deque
from math import sqrt

import numpy as np
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from gazetype.blink import DeliberateBlinkDetector
from gazetype.calibration import CalibrationAdapterModel, CalibrationModel, calibration_targets
from gazetype.gaze_model import (
    build_direct_general_predictor,
    build_runtime_predictor,
    configured_general_model_path,
    load_general_predictor,
)
from gazetype.input import create_input_sender
from gazetype.keyboards import KeyboardGeometry
from gazetype.landing import LandingDetector
from gazetype.models import GazePoint, KeyboardLayout, SENSITIVITY_PROFILES, Sensitivity, VisionFrame
from gazetype.settings import AppSettings, SettingsStore
from gazetype.training_data import TrainingSampleWriter, default_training_samples_path
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


def stabilize_binocular_features(
    features: list[float], model: CalibrationModel
) -> list[float]:
    """Fuse both eyes in normalized space while preserving each eye's calibration."""
    stabilized = features.copy()
    for left_index, right_index in ((0, 2), (1, 3)):
        left_delta = (
            features[left_index] - model.feature_mean[left_index]
        ) / model.feature_scale[left_index]
        right_delta = (
            features[right_index] - model.feature_mean[right_index]
        ) / model.feature_scale[right_index]
        shared_delta = (left_delta + right_delta) / 2.0
        stabilized[left_index] = (
            model.feature_mean[left_index]
            + shared_delta * model.feature_scale[left_index]
        )
        stabilized[right_index] = (
            model.feature_mean[right_index]
            + shared_delta * model.feature_scale[right_index]
        )
    return stabilized


def adaptive_gaze_point(
    previous: tuple[float, float] | None, current: tuple[float, float]
) -> tuple[float, float]:
    """Smooth small landmark jitter while following deliberate saccades quickly."""
    if previous is None:
        return current
    distance = ((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5
    alpha = 0.18 + 0.72 * min(distance / 0.18, 1.0)
    return (
        previous[0] + alpha * (current[0] - previous[0]),
        previous[1] + alpha * (current[1] - previous[1]),
    )


def _mean_prediction_error(
    predictions: list[tuple[float, float]],
    targets: np.ndarray,
) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(predictions) - targets, axis=1)))


def validated_calibration_adapter(
    general,
    features,
    targets,
    robust: bool,
    improvement_margin: float = 0.95,
) -> CalibrationAdapterModel | None:
    feature_array = np.asarray(tuple(features), dtype=np.float64)
    target_array = np.asarray(tuple(targets), dtype=np.float64)
    if len(feature_array) < 45:
        base_predictions = [general.predict(tuple(feature)) for feature in feature_array]
        return CalibrationAdapterModel.fit(base_predictions, feature_array, target_array)

    validation_mask = np.zeros(len(feature_array), dtype=bool)
    validation_mask[::5] = True
    training_features = feature_array[~validation_mask]
    training_targets = target_array[~validation_mask]
    validation_features = feature_array[validation_mask]
    validation_targets = target_array[validation_mask]
    if len(training_features) < 20 or len(validation_features) < 4:
        return None

    baseline = CalibrationModel.fit(training_features, training_targets, robust=robust)
    train_base_predictions = [general.predict(tuple(feature)) for feature in training_features]
    adapter = CalibrationAdapterModel.fit(
        train_base_predictions, training_features, training_targets
    )
    baseline_error = _mean_prediction_error(
        [baseline.predict(tuple(feature)) for feature in validation_features],
        validation_targets,
    )
    adapter_error = _mean_prediction_error(
        [
            adapter.predict(general.predict(tuple(feature)), tuple(feature))
            for feature in validation_features
        ],
        validation_targets,
    )
    if adapter_error > baseline_error * improvement_margin:
        return None

    all_base_predictions = [general.predict(tuple(feature)) for feature in feature_array]
    return CalibrationAdapterModel.fit(all_base_predictions, feature_array, target_array)


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
        self.input_sender = create_input_sender()
        self.landing = LandingDetector(SENSITIVITY_PROFILES[self.settings.sensitivity])
        self.blink = DeliberateBlinkDetector()
        self.keyboard_enabled = False
        self.face_present = False
        self.screen = None
        self.recent_gaze: deque[tuple[float, float]] = deque(maxlen=3)
        self.filtered_gaze: tuple[float, float] | None = None
        self.previous_head_sample: tuple[int, tuple[float, ...]] | None = None
        self.head_motion_until_ms = 0
        self.gaze_predictor = None
        self.direct_general_model = False
        if self.settings.calibration is not None:
            self._refresh_gaze_predictor()

        self.settings_window.start_requested.connect(self.begin_calibration)
        self.settings_window.model_start_requested.connect(self.begin_general_model)
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
        self.direct_general_model = False
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
            binocular_stabilization=bool(values["binocular_stabilization"]),
            adaptive_gaze_filter=bool(values["adaptive_gaze_filter"]),
            robust_calibration=bool(values["robust_calibration"]),
            use_general_gaze_model=bool(values["use_general_gaze_model"]),
            general_gaze_model_path=self.settings.general_gaze_model_path,
            collect_training_samples=bool(values["collect_training_samples"]),
            collect_training_images=self.settings.collect_training_images,
            horizontal_gain_percent=int(values["horizontal_gain_percent"]),
            vertical_gain_percent=int(values["vertical_gain_percent"]),
            vertical_offset_percent=int(values["vertical_offset_percent"]),
            head_compensation_percent=int(values["head_compensation_percent"]),
            head_motion_threshold_percent=int(values["head_motion_threshold_percent"]),
        )
        self.recent_gaze = deque(maxlen=self.settings.gaze_average_count)
        self.filtered_gaze = None
        self.tracking_window.configure_tuning(self.settings)
        self.previous_head_sample = None
        self.head_motion_until_ms = 0
        self.gaze_predictor = None
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
        training_writer = (
            TrainingSampleWriter(
                default_training_samples_path(),
                self.settings.screen_geometry,
                self.settings.camera_index,
            )
            if self.settings.collect_training_samples
            else None
        )
        self.calibration_window.begin(self.screen, targets, calibration_keyboard, training_writer)

    def begin_general_model(self, values: dict[str, object]) -> None:
        screens = self.application.screens()
        screen_index = int(values["screen_index"])
        if screen_index >= len(screens):
            self.settings_window.set_status("Seçilen ekran artık bağlı değil.", True)
            self.settings_window.unlock()
            self._refresh_screens()
            return
        use_general_model = bool(values["use_general_gaze_model"])
        if not use_general_model:
            self.settings_window.set_status("Kişisel model için genel model ayarını açın.", True)
            self.settings_window.unlock()
            return
        predictor = build_direct_general_predictor(
            use_general_model,
            self.settings.general_gaze_model_path,
        )
        if predictor is None:
            self.settings_window.set_status(
                "Model bulunamadı. GAZETYPE_GENERAL_MODEL ile .npz dosyasını gösterin.",
                True,
            )
            self.settings_window.unlock()
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
            auto_gaze_gain=False,
            quadrilateral_eye_mapping=bool(values["quadrilateral_eye_mapping"]),
            binocular_stabilization=False,
            adaptive_gaze_filter=bool(values["adaptive_gaze_filter"]),
            robust_calibration=bool(values["robust_calibration"]),
            use_general_gaze_model=True,
            general_gaze_model_path=self.settings.general_gaze_model_path,
            collect_training_samples=bool(values["collect_training_samples"]),
            collect_training_images=self.settings.collect_training_images,
            horizontal_gain_percent=int(values["horizontal_gain_percent"]),
            vertical_gain_percent=int(values["vertical_gain_percent"]),
            vertical_offset_percent=int(values["vertical_offset_percent"]),
            head_compensation_percent=100,
            head_motion_threshold_percent=int(values["head_motion_threshold_percent"]),
            calibration=self.settings.calibration,
            calibration_adapter=self.settings.calibration_adapter,
        )
        self.direct_general_model = True
        self.gaze_predictor = predictor
        self.recent_gaze = deque(maxlen=self.settings.gaze_average_count)
        self.filtered_gaze = None
        self.tracking_window.configure_tuning(self.settings)
        self.previous_head_sample = None
        self.head_motion_until_ms = 0
        self.landing = LandingDetector(SENSITIVITY_PROFILES[self.settings.sensitivity])
        self._stop_worker()
        self.worker = CameraWorker(
            self.settings.camera_index, self.settings.quadrilateral_eye_mapping
        )
        self.worker.frame_ready.connect(self.on_vision_frame)
        self.worker.tracking_preview.connect(self.tracking_window.set_frame)
        self.worker.face_presence.connect(self.on_face_presence)
        self.worker.error.connect(self.on_camera_error)
        self.worker.start()
        self.overlay.configure(self.screen, self.settings.layout)
        self.toggle.place(self.screen)
        self.keyboard_enabled = False
        self.toggle.set_enabled(False)
        self.tracking_window.show()
        self.settings_window.showMinimized()
        self.settings_window.unlock()

    def finish_calibration(self, features, targets) -> None:
        try:
            self.settings.calibration = CalibrationModel.fit(
                features, targets, robust=self.settings.robust_calibration
            )
            self.settings.calibration_adapter = None
            if self.settings.use_general_gaze_model:
                general = load_general_predictor(
                    configured_general_model_path(self.settings.general_gaze_model_path)
                )
                if general is not None:
                    self.settings.calibration_adapter = validated_calibration_adapter(
                        general,
                        features,
                        targets,
                        self.settings.robust_calibration,
                    )
            self.store.save(self.settings)
            self._refresh_gaze_predictor()
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
        model = None if self.direct_general_model else self.settings.calibration
        if self.screen is None:
            return
        compensated_features = list(frame.features)
        if model is not None:
            compensation = self.settings.head_compensation_percent / 100.0
            for index in range(4, len(compensated_features)):
                baseline = model.feature_mean[index]
                compensated_features[index] = baseline + (
                    compensated_features[index] - baseline
                ) * compensation
        if model is not None and self.settings.binocular_stabilization:
            compensated_features = stabilize_binocular_features(compensated_features, model)
        if self.gaze_predictor is None:
            self._refresh_gaze_predictor()
        if self.gaze_predictor is None:
            return
        x, y = self.gaze_predictor.predict(tuple(compensated_features))
        if model is not None and self.settings.auto_gaze_gain:
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
        if self.settings.adaptive_gaze_filter:
            self.filtered_gaze = adaptive_gaze_point(self.filtered_gaze, (x, y))
            x, y = self.filtered_gaze
        else:
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
            self.filtered_gaze = None
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

    def _refresh_gaze_predictor(self) -> None:
        if self.direct_general_model:
            self.gaze_predictor = build_direct_general_predictor(
                self.settings.use_general_gaze_model,
                self.settings.general_gaze_model_path,
            )
            return
        if self.settings.calibration is None:
            self.gaze_predictor = None
            return
        self.gaze_predictor = build_runtime_predictor(
            self.settings.calibration,
            self.settings.use_general_gaze_model,
            self.settings.general_gaze_model_path,
            self.settings.calibration_adapter,
        )

    def on_camera_error(self, message: str) -> None:
        self.calibration_window.hide()
        self.tracking_window.hide()
        self.settings_window.unlock()
        self.settings_window.set_status(message, True)
        self.settings_window.show()
        show_error(self.settings_window, "Kamera hatası", message)

    def toggle_keyboard(self) -> None:
        if (self.settings.calibration is None and not self.direct_general_model) or self.screen is None:
            return
        self.keyboard_enabled = not self.keyboard_enabled
        self.landing.reset()
        self.recent_gaze.clear()
        self.filtered_gaze = None
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
