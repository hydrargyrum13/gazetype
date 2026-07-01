import numpy as np

from gazetype.calibration import CALIBRATION_TARGETS, CalibrationModel, calibration_targets


def test_calibration_round_trip() -> None:
    features = [
        (x - 0.01, y, x + 0.01, y, 0.1 * x, 0.1 * y, 0.0, 0.3, 0.0, 0.0)
        for x, y in CALIBRATION_TARGETS
    ]
    model = CalibrationModel.fit(features, CALIBRATION_TARGETS)
    predicted = model.predict((0.49, 0.5, 0.51, 0.5, 0.05, 0.05, 0.0, 0.3, 0.0, 0.0))
    assert np.allclose(predicted, (0.5, 0.5), atol=0.03)
    restored = CalibrationModel.from_dict(model.to_dict())
    assert np.allclose(
        restored.predict((0.19, 0.8, 0.21, 0.8, 0.02, 0.08, 0.0, 0.3, 0.0, 0.0)),
        (0.2, 0.8), atol=0.05,
    )


def test_calibration_uses_twenty_five_point_snake_grid() -> None:
    assert len(CALIBRATION_TARGETS) == 25
    assert len(set(CALIBRATION_TARGETS)) == 25
    assert np.allclose(CALIBRATION_TARGETS[:5], tuple((x, 0.06) for x in (0.06, 0.28, 0.50, 0.72, 0.94)))
    assert np.allclose(CALIBRATION_TARGETS[5:10], tuple((x, 0.28) for x in (0.94, 0.72, 0.50, 0.28, 0.06)))


def test_calibration_target_count_is_configurable() -> None:
    assert len(calibration_targets(20)) == 20
    assert len(calibration_targets(37)) == 37
    assert len(set(calibration_targets(49))) == 49
    assert len(calibration_targets(81)) == 81


def test_calibration_rejects_too_few_samples() -> None:
    try:
        CalibrationModel.fit([(0,) * 10] * 4, [(0, 0)] * 4)
    except ValueError as error:
        assert "at least 20" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_old_calibration_model_is_rejected() -> None:
    try:
        CalibrationModel.from_dict({"coefficients": [[0.0] * 8, [0.0] * 8]})
    except ValueError as error:
        assert "outdated" in str(error)
    else:
        raise AssertionError("Expected ValueError")
