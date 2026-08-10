from PySide6.QtCore import QPoint

from gazetype.dataset_collector import (
    ScreenGeometry,
    guided_target_sequence,
    normalized_target_for_position,
)


def test_normalized_target_for_position_uses_screen_geometry() -> None:
    geometry = ScreenGeometry(10, 20, 200, 100)

    assert normalized_target_for_position(QPoint(110, 70), geometry) == (0.5, 0.5)


def test_normalized_target_for_position_clamps_to_screen() -> None:
    geometry = ScreenGeometry(10, 20, 200, 100)

    assert normalized_target_for_position(QPoint(-90, 220), geometry) == (0.0, 1.0)


def test_guided_target_sequence_covers_screen_area() -> None:
    targets = guided_target_sequence(25, seed=4)
    xs = [target[0] for target in targets]
    ys = [target[1] for target in targets]

    assert len(targets) == 25
    assert min(xs) <= 0.06
    assert max(xs) >= 0.94
    assert min(ys) <= 0.07
    assert max(ys) >= 0.93
