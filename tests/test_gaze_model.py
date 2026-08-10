import numpy as np

from gazetype.app import validated_calibration_adapter
from gazetype.calibration import CalibrationAdapterModel, CalibrationModel
from gazetype.gaze_model import build_runtime_predictor


def _features(count: int = 25) -> np.ndarray:
    xs = np.linspace(0.1, 0.9, count)
    return np.asarray([
        (x - 0.01, 0.5, x + 0.01, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0)
        for x in xs
    ])


def test_general_model_missing_falls_back_to_calibration(tmp_path) -> None:
    features = _features()
    targets = np.column_stack((np.linspace(0.1, 0.9, len(features)), np.full(len(features), 0.5)))
    calibration = CalibrationModel.fit(features, targets)
    adapter = CalibrationAdapterModel.fit(targets, features, targets)
    predictor = build_runtime_predictor(
        calibration,
        True,
        str(tmp_path / "missing.npz"),
        adapter,
    )
    assert np.allclose(predictor.predict(tuple(features[8])), calibration.predict(features[8]))


class ConstantPredictor:
    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        return (0.5, 0.5)


class UsefulPredictor:
    def predict(self, features: tuple[float, ...]) -> tuple[float, float]:
        return (
            float(np.clip(0.5 + features[8] * features[9] * 2.0, 0.0, 1.0)),
            float(np.clip(0.5 - features[8] * features[9] * 2.0, 0.0, 1.0)),
        )


def test_general_adapter_is_rejected_when_holdout_is_worse_than_calibration() -> None:
    rng = np.random.default_rng(21)
    features = rng.normal(0.0, 0.2, size=(80, 10))
    targets = np.column_stack((
        np.clip(0.5 + features[:, 0] * 0.7, 0.0, 1.0),
        np.clip(0.5 + features[:, 1] * 0.7, 0.0, 1.0),
    ))

    adapter = validated_calibration_adapter(ConstantPredictor(), features, targets, robust=True)
    assert adapter is None


def test_general_adapter_is_kept_when_it_improves_holdout() -> None:
    rng = np.random.default_rng(22)
    features = rng.normal(0.0, 0.25, size=(120, 10))
    targets = np.column_stack((
        np.clip(0.5 + features[:, 8] * features[:, 9] * 2.0, 0.0, 1.0),
        np.clip(0.5 - features[:, 8] * features[:, 9] * 2.0, 0.0, 1.0),
    ))

    adapter = validated_calibration_adapter(UsefulPredictor(), features, targets, robust=True)
    assert adapter is not None
