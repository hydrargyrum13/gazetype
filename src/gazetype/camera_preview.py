from __future__ import annotations

import sys

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


class CameraPreviewWorker(QThread):
    """Reads a low-rate camera preview without blocking the Qt event loop."""

    frame_ready = Signal(int, object)
    availability_changed = Signal(int, bool)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(2000)

    def run(self) -> None:
        import cv2

        # Missing camera indices are expected during discovery; report them in
        # the UI instead of printing OpenCV backend warnings to the console.
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(2)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.camera_index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
        capture.set(cv2.CAP_PROP_FPS, 10)
        if not capture.isOpened():
            self.availability_changed.emit(self.camera_index, False)
            capture.release()
            return
        self.availability_changed.emit(self.camera_index, True)
        try:
            while not self.isInterruptionRequested():
                success, frame = capture.read()
                if not success:
                    self.msleep(80)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(self.camera_index, image)
                self.msleep(70)
        finally:
            capture.release()
