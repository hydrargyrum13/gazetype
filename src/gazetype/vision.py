from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from gazetype.models import VisionFrame


def model_path() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    candidates = []
    if frozen_root:
        candidates.append(Path(frozen_root) / "gazetype" / "assets" / "face_landmarker.task")
    candidates.append(Path(__file__).resolve().parent / "assets" / "face_landmarker.task")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "face_landmarker.task bulunamadı. scripts/download_model.ps1 komutunu çalıştırın."
    )


def _normalized(value: float, first: float, second: float) -> float:
    low, high = sorted((first, second))
    return (value - low) / max(high - low, 1e-6)


def extract_gaze_features(landmarks: object) -> tuple[float, float, float, float]:
    """Extract iris position and coarse head offset from 478 face landmarks."""
    points = landmarks
    left_iris_x = float(np.mean([points[index].x for index in range(468, 473)]))
    left_iris_y = float(np.mean([points[index].y for index in range(468, 473)]))
    right_iris_x = float(np.mean([points[index].x for index in range(473, 478)]))
    right_iris_y = float(np.mean([points[index].y for index in range(473, 478)]))

    left_x = _normalized(left_iris_x, points[33].x, points[133].x)
    right_x = _normalized(right_iris_x, points[362].x, points[263].x)
    left_y = _normalized(left_iris_y, points[159].y, points[145].y)
    right_y = _normalized(right_iris_y, points[386].y, points[374].y)

    eye_mid_x = (points[33].x + points[263].x) / 2
    eye_mid_y = (points[159].y + points[386].y) / 2
    eye_distance = max(abs(points[263].x - points[33].x), 1e-6)
    head_x = (points[1].x - eye_mid_x) / eye_distance
    head_y = (points[1].y - eye_mid_y) / eye_distance
    return (
        float(np.clip((left_x + right_x) / 2, -0.5, 1.5)),
        float(np.clip((left_y + right_y) / 2, -0.5, 1.5)),
        float(np.clip(head_x, -1.0, 1.0)),
        float(np.clip(head_y, -1.0, 1.0)),
    )


class CameraWorker(QThread):
    frame_ready = Signal(object)
    tracking_preview = Signal(object)
    face_presence = Signal(bool)
    error = Signal(str)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(2500)

    def run(self) -> None:  # noqa: C901 - camera lifecycle is intentionally kept in one thread
        try:
            import cv2
            import mediapipe as mp

            path = model_path()
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.camera_index, backend)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            if not capture.isOpened():
                self.error.emit(f"Kamera {self.camera_index} açılamadı.")
                return

            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
            previous_presence: bool | None = None
            previous_time = time.perf_counter()
            smoothed_fps = 30.0
            started = time.perf_counter()
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while not self.isInterruptionRequested():
                    success, image = capture.read()
                    if not success:
                        self.msleep(10)
                        continue
                    image = cv2.flip(image, 1)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    timestamp_ms = int((time.perf_counter() - started) * 1000)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(mp_image, timestamp_ms)
                    present = bool(result.face_landmarks)
                    if present != previous_presence:
                        self.face_presence.emit(present)
                        previous_presence = present
                    now = time.perf_counter()
                    instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                    smoothed_fps = 0.9 * smoothed_fps + 0.1 * instantaneous_fps
                    previous_time = now
                    if present:
                        points = result.face_landmarks[0]
                        height, width = image.shape[:2]
                        eye_color = (151, 201, 44)
                        for contour in (
                            (33, 160, 158, 133, 153, 144),
                            (362, 385, 387, 263, 373, 380),
                        ):
                            polygon = np.asarray([
                                (int(points[index].x * width), int(points[index].y * height))
                                for index in contour
                            ], dtype=np.int32)
                            cv2.polylines(image, [polygon], True, eye_color, 2, cv2.LINE_AA)
                        for iris_range in (range(468, 473), range(473, 478)):
                            iris_x = int(np.mean([points[index].x for index in iris_range]) * width)
                            iris_y = int(np.mean([points[index].y for index in iris_range]) * height)
                            cv2.circle(image, (iris_x, iris_y), 6, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(image, (iris_x, iris_y), 2, eye_color, -1, cv2.LINE_AA)
                        xs = [point.x for point in points[:468]]
                        ys = [point.y for point in points[:468]]
                        top_left = (int(min(xs) * width), int(min(ys) * height))
                        bottom_right = (int(max(xs) * width), int(max(ys) * height))
                        cv2.rectangle(image, top_left, bottom_right, (151, 201, 44), 1, cv2.LINE_AA)
                    rgb_preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    height, width, channels = rgb_preview.shape
                    preview = QImage(
                        rgb_preview.data,
                        width,
                        height,
                        channels * width,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.tracking_preview.emit(preview)
                    if not present:
                        continue
                    features = extract_gaze_features(result.face_landmarks[0])
                    blendshapes = {
                        category.category_name: float(category.score)
                        for category in result.face_blendshapes[0]
                    } if result.face_blendshapes else {}
                    self.frame_ready.emit(VisionFrame(
                        timestamp_ms=timestamp_ms,
                        features=features,
                        blink_left=blendshapes.get("eyeBlinkLeft", 0.0),
                        blink_right=blendshapes.get("eyeBlinkRight", 0.0),
                        fps=smoothed_fps,
                    ))
            capture.release()
        except Exception as exc:  # errors must reach the GUI instead of killing the worker silently
            self.error.emit(str(exc))
