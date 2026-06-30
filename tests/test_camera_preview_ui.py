import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from gazetype.settings import AppSettings
from gazetype.ui import SettingsWindow


def test_camera_preview_cards_select_camera() -> None:
    application = QApplication.instance() or QApplication([])
    window = SettingsWindow(AppSettings(camera_index=1))
    assert len(window.camera_cards) == 4
    assert window.selected_camera_index == 1
    window.select_camera(3)
    assert window.selected_camera_index == 3
    assert window.camera_cards[3].isChecked()
    assert not window.camera_cards[1].isChecked()
    window.stop_camera_previews()
    window.close()
    assert application is not None

