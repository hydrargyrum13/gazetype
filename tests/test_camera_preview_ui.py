import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from gazetype.settings import AppSettings
from gazetype.ui import KeyboardOverlay, SettingsWindow, TrackingWindow


def test_camera_preview_cards_select_camera() -> None:
    application = QApplication.instance() or QApplication([])
    window = SettingsWindow(AppSettings(camera_index=1))
    assert len(window.camera_cards) == 4
    assert window.selected_camera_index == 1
    window.select_camera(3)
    assert window.selected_camera_index == 3
    assert window.camera_cards[3].isChecked()
    assert not window.camera_cards[1].isChecked()
    assert window.settings_tabs.count() == 2
    assert window.settings_tabs.tabText(0) == "Ayarlar"
    assert window.settings_tabs.tabText(1) == "Gelişmiş Ayarlar"
    assert window.settings_tabs.currentIndex() == 0
    window.calibration_mode.setCurrentIndex(1)
    assert window.calibration_mode.currentData() == "keyboard"
    assert not window.point_count.isEnabled()
    window.stop_camera_previews()
    window.close()
    assert application is not None


def test_overlay_tracks_visible_gaze_position() -> None:
    application = QApplication.instance() or QApplication([])
    overlay = KeyboardOverlay()
    overlay.set_gaze_state("a", 0.5, 30.0, (0.25, 0.75))
    assert overlay.gaze_position == (0.25, 0.75)
    assert overlay.keyboard.top == 0.03
    overlay.close()
    assert application is not None


def test_tracking_window_displays_camera_frame() -> None:
    application = QApplication.instance() or QApplication([])
    window = TrackingWindow()
    image = QImage(640, 480, QImage.Format.Format_RGB888)
    image.fill(QColor("black"))
    window.set_frame(image)
    assert window.preview.pixmap() is not None
    window.close()
    assert application is not None
