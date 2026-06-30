from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gazetype.calibration import CALIBRATION_TARGETS
from gazetype.camera_preview import CameraPreviewWorker
from gazetype.keyboards import KeyboardGeometry
from gazetype.models import KeyboardLayout, Sensitivity
from gazetype.settings import AppSettings
from gazetype.windows import make_window_non_activating


COLORS = {
    "background": QColor(10, 14, 23, 225),
    "key": QColor(32, 40, 55, 242),
    "key_border": QColor(105, 124, 153),
    "active": QColor(44, 201, 151),
    "text": QColor(246, 248, 252),
    "muted": QColor(174, 186, 204),
    "danger": QColor(239, 91, 91),
}


def _ui_font(pixel_size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont("Segoe UI")
    font.setPixelSize(max(1, pixel_size))
    font.setWeight(weight)
    return font


class CameraPreviewCard(QPushButton):
    selected = Signal(int)

    def __init__(self, camera_index: int):
        super().__init__()
        self.camera_index = camera_index
        self.setCheckable(True)
        self.setMinimumSize(188, 142)
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.preview = QLabel("Kamera aranıyor…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(170, 96)
        self.preview.setStyleSheet("background: #080b11; color: #aebacc; border-radius: 5px;")
        self.caption = QLabel(f"Kamera {camera_index + 1}")
        self.caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.caption)
        self.clicked.connect(lambda: self.selected.emit(self.camera_index))
        self.set_available(None)

    def set_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setPixmap(pixmap)
        self.preview.setText("")

    def set_available(self, available: bool | None) -> None:
        if available is None:
            self.caption.setText(f"Kamera {self.camera_index + 1} · aranıyor")
            self.setEnabled(True)
        elif available:
            self.caption.setText(f"Kamera {self.camera_index + 1} · hazır")
            self.setEnabled(True)
        else:
            self.preview.clear()
            self.preview.setText("Kullanılamıyor")
            self.caption.setText(f"Kamera {self.camera_index + 1} · bulunamadı")
            self.setEnabled(False)
        self._refresh_style()

    def set_selected(self, selected: bool) -> None:
        self.setChecked(selected)
        self._refresh_style()

    def _refresh_style(self) -> None:
        border = "#2cc997" if self.isChecked() else "#526078"
        self.setStyleSheet(
            f"QPushButton {{ background: #202837; border: 3px solid {border}; border-radius: 9px; }}"
            "QPushButton:hover { border-color: #2cc997; }"
        )


class SettingsWindow(QMainWindow):
    start_requested = Signal(object)

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.setWindowTitle("Gazetype")
        self.setMinimumSize(520, 620)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("Gazetype")
        title.setStyleSheet("font-size: 28px; font-weight: 700;")
        subtitle = QLabel("Gözle yazma klavyesini kalibre edin ve başlatın.")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        camera_title = QLabel("Kamera önizlemeleri")
        camera_title.setStyleSheet("font-weight: 700; margin-top: 8px;")
        layout.addWidget(camera_title)
        camera_grid = QGridLayout()
        camera_grid.setSpacing(8)
        self.selected_camera_index = max(0, min(settings.camera_index, 3))
        self.camera_cards: dict[int, CameraPreviewCard] = {}
        self.camera_availability: dict[int, bool | None] = {}
        self.preview_workers: dict[int, CameraPreviewWorker] = {}
        for index in range(4):
            card = CameraPreviewCard(index)
            card.selected.connect(self.select_camera)
            card.set_selected(index == self.selected_camera_index)
            self.camera_cards[index] = card
            self.camera_availability[index] = None
            camera_grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(camera_grid)

        form = QFormLayout()

        self.screen_combo = QComboBox()
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Türkçe Q", KeyboardLayout.TURKISH_Q)
        self.layout_combo.addItem("English QWERTY", KeyboardLayout.ENGLISH_QWERTY)
        self.layout_combo.setCurrentIndex(0 if settings.layout == KeyboardLayout.TURKISH_Q else 1)

        self.sensitivity_combo = QComboBox()
        for label, value in (
            ("Hızlı — 2 kare / 50 ms", Sensitivity.FAST),
            ("Dengeli — 3 kare / 90 ms", Sensitivity.BALANCED),
            ("Sabit — 4 kare / 130 ms", Sensitivity.STABLE),
        ):
            self.sensitivity_combo.addItem(label, value)
        self.sensitivity_combo.setCurrentIndex(list(Sensitivity).index(settings.sensitivity))
        form.addRow("Ekran", self.screen_combo)
        form.addRow("Klavye", self.layout_combo)
        form.addRow("Hassasiyet", self.sensitivity_combo)
        layout.addLayout(form)

        self.status = QLabel("Hazır")
        self.status.setWordWrap(True)
        self.start_button = QPushButton("9 Noktalı Kalibrasyonu Başlat")
        self.start_button.setMinimumHeight(44)
        self.start_button.clicked.connect(self._emit_start)
        layout.addWidget(self.status)
        layout.addWidget(self.start_button)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #111722; color: #f6f8fc; font-size: 14px; }"
            "QComboBox, QPushButton { padding: 8px; background: #202837; border: 1px solid #526078; border-radius: 6px; }"
            "QPushButton:hover { border-color: #2cc997; }"
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.start_camera_previews()

    def hideEvent(self, event) -> None:
        self.stop_camera_previews()
        super().hideEvent(event)

    def start_camera_previews(self) -> None:
        if self.preview_workers:
            return
        for index, card in self.camera_cards.items():
            self.camera_availability[index] = None
            card.set_available(None)
            worker = CameraPreviewWorker(index, self)
            worker.frame_ready.connect(self._set_camera_frame)
            worker.availability_changed.connect(self._set_camera_availability)
            self.preview_workers[index] = worker
            worker.start()

    def stop_camera_previews(self) -> None:
        workers = tuple(self.preview_workers.values())
        self.preview_workers.clear()
        for worker in workers:
            worker.stop()
            worker.deleteLater()

    def select_camera(self, camera_index: int) -> None:
        self.selected_camera_index = camera_index
        for index, card in self.camera_cards.items():
            card.set_selected(index == camera_index)

    def _set_camera_frame(self, camera_index: int, image: QImage) -> None:
        card = self.camera_cards.get(camera_index)
        if card:
            card.set_frame(image)

    def _set_camera_availability(self, camera_index: int, available: bool) -> None:
        card = self.camera_cards.get(camera_index)
        if not card:
            return
        self.camera_availability[camera_index] = available
        card.set_available(available)
        if not available and camera_index == self.selected_camera_index:
            available_indices = [index for index, state in self.camera_availability.items() if state is True]
            if available_indices:
                self.select_camera(available_indices[0])
        elif available and self.camera_availability.get(self.selected_camera_index) is False:
            self.select_camera(camera_index)

    def set_screens(self, screens: list[tuple[str, str]], selected_name: str) -> None:
        self.screen_combo.clear()
        selected = 0
        for index, (label, geometry) in enumerate(screens):
            self.screen_combo.addItem(label, (label, geometry, index))
            if label == selected_name:
                selected = index
        self.screen_combo.setCurrentIndex(selected)

    def _emit_start(self) -> None:
        self.start_button.setEnabled(False)
        self.status.setText("Kamera başlatılıyor…")
        screen_name, geometry, screen_index = self.screen_combo.currentData()
        self.stop_camera_previews()
        self.start_requested.emit({
            "camera_index": self.selected_camera_index,
            "screen_name": screen_name,
            "screen_geometry": geometry,
            "screen_index": screen_index,
            "layout": self.layout_combo.currentData(),
            "sensitivity": self.sensitivity_combo.currentData(),
        })

    def set_status(self, message: str, error: bool = False) -> None:
        color = "#ef5b5b" if error else "#aebacc"
        self.status.setStyleSheet(f"color: {color};")
        self.status.setText(message)

    def unlock(self) -> None:
        self.start_button.setEnabled(True)


class CalibrationWindow(QWidget):
    completed = Signal(object, object)
    cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setCursor(Qt.BlankCursor)
        self._target_index = 0
        self._target_started_ms: int | None = None
        self._samples: list[tuple[float, float, float, float]] = []
        self._collected: list[tuple[float, float, float, float]] = []
        self._face_present = False

    def begin(self, screen) -> None:
        self.setGeometry(screen.geometry())
        self._target_index = 0
        self._target_started_ms = None
        self._samples.clear()
        self._collected.clear()
        self.show()
        self.raise_()
        self.update()

    def set_face_present(self, present: bool) -> None:
        self._face_present = present
        if not present:
            self._target_started_ms = None
            self._samples.clear()
        self.update()

    def add_sample(self, timestamp_ms: int, features: tuple[float, float, float, float]) -> None:
        if not self.isVisible() or not self._face_present:
            return
        if self._target_started_ms is None:
            self._target_started_ms = timestamp_ms
            return
        if timestamp_ms - self._target_started_ms < 260:
            return
        self._samples.append(features)
        if len(self._samples) < 12:
            self.update()
            return
        median = tuple(float(value) for value in np.median(np.asarray(self._samples), axis=0))
        self._collected.append(median)
        self._samples.clear()
        self._target_started_ms = None
        self._target_index += 1
        if self._target_index >= len(CALIBRATION_TARGETS):
            self.hide()
            self.completed.emit(tuple(self._collected), CALIBRATION_TARGETS)
        else:
            self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.hide()
            self.cancelled.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(10, 14, 23))
        target = CALIBRATION_TARGETS[min(self._target_index, 8)]
        point = QPointF(target[0] * self.width(), target[1] * self.height())
        painter.setPen(Qt.NoPen)
        painter.setBrush(COLORS["active"] if self._face_present else COLORS["danger"])
        painter.drawEllipse(point, 19, 19)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(point, 6, 6)
        painter.setPen(COLORS["text"])
        painter.setFont(_ui_font(24, QFont.Weight.Bold))
        painter.drawText(
            QRectF(0, self.height() * 0.43, self.width(), 50),
            Qt.AlignCenter,
            f"Noktaya bakın  •  {self._target_index + 1}/9",
        )
        painter.setPen(COLORS["muted"])
        painter.setFont(_ui_font(16))
        message = "Yüz algılandı" if self._face_present else "Yüzünüzü kameraya gösterin"
        painter.drawText(QRectF(0, self.height() * 0.49, self.width(), 40), Qt.AlignCenter, message)


class KeyboardOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.keyboard = KeyboardGeometry(KeyboardLayout.TURKISH_Q)
        self.active_key: str | None = None
        self.progress = 0.0
        self.face_present = True
        self.fps = 0.0

    def configure(self, screen, layout: KeyboardLayout) -> None:
        self.keyboard = KeyboardGeometry(layout)
        self.setGeometry(screen.geometry())

    def show_non_activating(self) -> None:
        self.show()
        make_window_non_activating(int(self.winId()), click_through=True)

    def set_gaze_state(self, key_id: str | None, progress: float, fps: float) -> None:
        self.active_key = key_id
        self.progress = progress
        self.fps = fps
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        keyboard_top = int(self.keyboard.top * self.height()) - 18
        painter.fillRect(QRect(0, keyboard_top, self.width(), self.height() - keyboard_top), COLORS["background"])
        for key in self.keyboard.keys:
            rect = QRectF(
                key.x * self.width(), key.y * self.height(), key.width * self.width(), key.height * self.height()
            )
            active = key.id == self.active_key
            painter.setBrush(COLORS["key"])
            painter.setPen(QPen(COLORS["active"] if active else COLORS["key_border"], 4 if active else 1))
            painter.drawRoundedRect(rect, 8, 8)
            if active and self.progress > 0:
                fill = QRectF(rect.x(), rect.bottom() - rect.height() * self.progress, rect.width(), rect.height() * self.progress)
                painter.save()
                painter.setClipPath(self._rounded_path(rect))
                painter.fillRect(fill, QColor(44, 201, 151, 75))
                painter.restore()
            painter.setPen(COLORS["text"])
            font_size = max(12, min(24, int(rect.height() * 0.32)))
            painter.setFont(_ui_font(font_size, QFont.Weight.DemiBold))
            painter.drawText(rect, Qt.AlignCenter, key.label.upper() if len(key.label) == 1 else key.label)
        painter.setPen(COLORS["muted"] if self.face_present else COLORS["danger"])
        status = f"Kamera {self.fps:.0f} FPS" if self.face_present else "Yüz algılanamadı — yazma duraklatıldı"
        painter.setFont(_ui_font(13))
        painter.drawText(QRectF(16, keyboard_top, 400, 18), Qt.AlignVCenter, status)

    @staticmethod
    def _rounded_path(rect: QRectF):
        from PySide6.QtGui import QPainterPath

        path = QPainterPath()
        path.addRoundedRect(rect, 8, 8)
        return path


class ToggleWindow(QWidget):
    toggled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        self.button = QPushButton("GÖZ KLAVYESİ\nKAPALI")
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.clicked.connect(self.toggled)
        layout.addWidget(self.button)
        self.set_enabled(False)

    def place(self, screen) -> None:
        geometry = screen.geometry()
        self.setGeometry(geometry.right() - 144, geometry.top() + 18, 126, 70)
        self.show()
        make_window_non_activating(int(self.winId()), click_through=False)

    def set_enabled(self, enabled: bool) -> None:
        self.button.setText("GÖZ KLAVYESİ\nAÇIK" if enabled else "GÖZ KLAVYESİ\nKAPALI")
        color = "#2cc997" if enabled else "#ef5b5b"
        self.button.setStyleSheet(
            f"QPushButton {{ color: white; background: #111722; border: 3px solid {color};"
            " border-radius: 12px; font-size: 11px; font-weight: 700; }}"
        )


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)
