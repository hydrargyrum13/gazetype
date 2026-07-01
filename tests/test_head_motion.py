from gazetype.app import head_motion_speed


def test_head_motion_speed_ignores_steady_pose() -> None:
    features = (0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.0, 0.3, 0.0, 0.0)
    assert head_motion_speed((0, features), 33, features) == 0.0


def test_head_motion_speed_detects_rapid_pose_change() -> None:
    first = (0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.0, 0.3, 0.0, 0.0)
    second = (0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.08, 0.34, 0.08, 0.08)
    assert head_motion_speed((0, first), 33, second) > 0.9
