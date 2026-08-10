from PySide6.QtCore import QPoint

from gazetype.dataset_collector import ScreenGeometry, normalized_target_for_position


def test_normalized_target_for_position_uses_screen_geometry() -> None:
    geometry = ScreenGeometry(10, 20, 200, 100)

    assert normalized_target_for_position(QPoint(110, 70), geometry) == (0.5, 0.5)


def test_normalized_target_for_position_clamps_to_screen() -> None:
    geometry = ScreenGeometry(10, 20, 200, 100)

    assert normalized_target_for_position(QPoint(-90, 220), geometry) == (0.0, 1.0)
