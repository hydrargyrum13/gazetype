import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from gazetype.calibration import CALIBRATION_TARGETS
from gazetype.ui import CalibrationWindow


def test_calibration_window_collects_all_twenty_five_points() -> None:
    application = QApplication.instance() or QApplication([])
    window = CalibrationWindow()
    completed: list[tuple[object, object]] = []
    window.completed.connect(lambda features, targets: completed.append((features, targets)))
    window.show()
    window.set_face_present(True)

    timestamp = 0
    for target_index in range(len(CALIBRATION_TARGETS)):
        features = (
            target_index / 25, 0.5, target_index / 25, 0.5,
            0.0, 0.0, 0.0, 0.3, 0.0, 0.0,
        )
        window.add_sample(timestamp, features)
        for sample_index in range(10):
            window.add_sample(timestamp + 201 + sample_index * 33, features)
        QTest.mouseClick(window, Qt.LeftButton)
        timestamp += 600

    assert len(completed) == 1
    features, targets = completed[0]
    assert len(features) == 200
    assert len(targets) == 200
    assert targets[0] == CALIBRATION_TARGETS[0]
    assert targets[-1] == CALIBRATION_TARGETS[-1]
    assert not window.isVisible()
    assert application is not None
