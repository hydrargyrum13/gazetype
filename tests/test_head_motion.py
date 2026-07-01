from gazetype.app import (
    adaptive_gaze_point,
    eye_ratio_gains,
    head_motion_speed,
    stabilize_binocular_features,
)
from gazetype.calibration import CalibrationModel


def test_head_motion_speed_ignores_steady_pose() -> None:
    features = (0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.0, 0.3, 0.0, 0.0)
    assert head_motion_speed((0, features), 33, features) == 0.0


def test_head_motion_speed_detects_rapid_pose_change() -> None:
    first = (0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.0, 0.3, 0.0, 0.0)
    second = (0.5, 0.5, 0.5, 0.5, 0.2, 0.3, 0.08, 0.34, 0.08, 0.08)
    assert head_motion_speed((0, first), 33, second) > 0.9


def test_eye_ratio_gains_boost_the_axis_with_less_calibration_movement() -> None:
    model = CalibrationModel(
        coefficients=((0.0,) * 36, (0.0,) * 36),
        feature_mean=(0.0,) * 10,
        feature_scale=(0.08, 0.02, 0.08, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02),
    )
    horizontal, vertical = eye_ratio_gains(model)
    assert horizontal == 50.0
    assert vertical == 200.0


def test_eye_ratio_gains_stay_neutral_for_equal_ratio_spreads() -> None:
    model = CalibrationModel(
        coefficients=((0.0,) * 36, (0.0,) * 36),
        feature_mean=(0.0,) * 10,
        feature_scale=(0.04,) * 10,
    )
    assert eye_ratio_gains(model) == (100.0, 100.0)


def test_binocular_stabilization_averages_independent_eye_noise() -> None:
    model = CalibrationModel(
        coefficients=((0.0,) * 36, (0.0,) * 36),
        feature_mean=(0.5, 0.5, 0.6, 0.4, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
        feature_scale=(0.1,) * 10,
    )
    result = stabilize_binocular_features(
        [0.7, 0.4, 0.6, 0.6, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0], model
    )
    assert result[:4] == [0.6, 0.55, 0.7, 0.45]


def test_adaptive_filter_smooths_jitter_but_follows_large_movements() -> None:
    jittered = adaptive_gaze_point((0.5, 0.5), (0.51, 0.5))
    saccade = adaptive_gaze_point((0.5, 0.5), (0.9, 0.5))
    assert 0.5 < jittered[0] < 0.505
    assert saccade[0] > 0.85
