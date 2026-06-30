import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

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
        features = (target_index / 25, 0.5, 0.0, 0.0)
        window.add_sample(timestamp, features)
        for sample_index in range(10):
            window.add_sample(timestamp + 201 + sample_index * 33, features)
        timestamp += 600

    assert len(completed) == 1
    features, targets = completed[0]
    assert len(features) == 25
    assert targets == CALIBRATION_TARGETS
    assert not window.isVisible()
    assert application is not None

