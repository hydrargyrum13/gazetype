import numpy as np

from gazetype.calibration import (
    ADAPTER_BASIS_SIZE,
    CALIBRATION_TARGETS,
    CalibrationAdapterModel,
    CalibrationModel,
    calibration_targets,
)


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


def test_robust_calibration_reduces_the_effect_of_a_bad_sample() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(0.0, 0.15, size=(60, 10))
    targets = np.column_stack((
        0.5 + features[:, 0] * 0.8,
        0.5 + features[:, 1] * 0.8,
    ))
    corrupted_targets = targets.copy()
    corrupted_targets[0] = (1.0, 0.0)
    robust = CalibrationModel.fit(features, corrupted_targets, robust=True)
    ordinary = CalibrationModel.fit(features, corrupted_targets, robust=False)
    robust_error = np.linalg.norm(np.asarray(robust.predict(features[1])) - targets[1])
    ordinary_error = np.linalg.norm(np.asarray(ordinary.predict(features[1])) - targets[1])
    assert robust_error < ordinary_error


def test_calibration_adapter_round_trip_and_clamps_correction() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(0.0, 0.15, size=(40, 10))
    targets = np.column_stack((
        np.clip(0.5 + features[:, 0] * 0.4, 0.0, 1.0),
        np.clip(0.5 + features[:, 1] * 0.4, 0.0, 1.0),
    ))
    base = np.clip(targets + np.asarray((0.08, -0.06)), 0.0, 1.0)
    adapter = CalibrationAdapterModel.fit(base, features, targets, max_correction=0.1)
    restored = CalibrationAdapterModel.from_dict(adapter.to_dict())
    prediction = restored.predict(base[0], features[0])
    assert np.allclose(prediction, targets[0], atol=0.04)
    clamping_adapter = CalibrationAdapterModel(
        ((1.0,) + (0.0,) * (ADAPTER_BASIS_SIZE - 1), (-1.0,) + (0.0,) * (ADAPTER_BASIS_SIZE - 1)),
        (0.0,) * 12,
        (1.0,) * 12,
        0.1,
    )
    assert clamping_adapter.predict((0.0, 1.0), (0.0,) * 10) == (0.1, 0.9)


def test_ridge_correction_improves_synthetic_general_model_error() -> None:
    rng = np.random.default_rng(15)
    features = rng.normal(0.0, 0.2, size=(60, 10))
    targets = np.column_stack((
        np.clip(0.5 + features[:, 0] * 0.5 + features[:, 4] * 0.06, 0.0, 1.0),
        np.clip(0.5 + features[:, 1] * 0.5 + features[:, 5] * 0.06, 0.0, 1.0),
    ))
    base = np.clip(targets + np.column_stack((features[:, 4] * 0.2 + 0.05, -features[:, 5] * 0.2)), 0.0, 1.0)
    adapter = CalibrationAdapterModel.fit(base, features, targets)
    base_error = np.mean(np.linalg.norm(base - targets, axis=1))
    corrected = np.asarray([adapter.predict(base_row, feature) for base_row, feature in zip(base, features, strict=True)])
    corrected_error = np.mean(np.linalg.norm(corrected - targets, axis=1))
    assert corrected_error < base_error * 0.65
