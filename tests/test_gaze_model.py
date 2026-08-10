import numpy as np

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
