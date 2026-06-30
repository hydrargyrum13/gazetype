import numpy as np

from gazetype.calibration import CALIBRATION_TARGETS, CalibrationModel


def test_calibration_round_trip() -> None:
    features = [(x, y, 0.1 * x, 0.1 * y) for x, y in CALIBRATION_TARGETS]
    model = CalibrationModel.fit(features, CALIBRATION_TARGETS)
    predicted = model.predict((0.5, 0.5, 0.05, 0.05))
    assert np.allclose(predicted, (0.5, 0.5), atol=0.03)
    restored = CalibrationModel.from_dict(model.to_dict())
    assert np.allclose(restored.predict((0.2, 0.8, 0.02, 0.08)), (0.2, 0.8), atol=0.05)


def test_calibration_rejects_too_few_samples() -> None:
    try:
        CalibrationModel.fit([(0, 0, 0, 0)] * 4, [(0, 0)] * 4)
    except ValueError as error:
        assert "at least eight" in str(error)
    else:
        raise AssertionError("Expected ValueError")

