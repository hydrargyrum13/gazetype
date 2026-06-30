from gazetype.landing import LandingDetector
from gazetype.models import GazePoint, SENSITIVITY_PROFILES, Sensitivity


def test_fast_saccade_selects_only_landing_key() -> None:
    detector = LandingDetector(SENSITIVITY_PROFILES[Sensitivity.FAST])
    samples = [
        (0, 0.10, 0.70, "a"),
        (33, 0.30, 0.70, "s"),
        (66, 0.70, 0.70, "j"),
        (99, 0.705, 0.702, "j"),
    ]
    selected = [
        detector.update(GazePoint(timestamp, x, y), key)
        for timestamp, x, y, key in samples
    ]
    assert selected == [None, None, None, "j"]


def test_selected_key_is_locked_until_gaze_leaves() -> None:
    detector = LandingDetector(SENSITIVITY_PROFILES[Sensitivity.FAST])
    detector.update(GazePoint(0, 0.1, 0.7), "a")
    detector.update(GazePoint(40, 0.6, 0.7), "h")
    assert detector.update(GazePoint(75, 0.6, 0.7), "h") == "h"
    assert detector.update(GazePoint(120, 0.6, 0.7), "h") is None
    detector.update(GazePoint(160, 0.9, 0.4), None)
    detector.update(GazePoint(200, 0.6, 0.7), "h")
    assert detector.update(GazePoint(235, 0.6, 0.7), "h") == "h"


def test_face_loss_reset_clears_state() -> None:
    detector = LandingDetector(SENSITIVITY_PROFILES[Sensitivity.FAST])
    detector.update(GazePoint(0, 0.1, 0.7), "a")
    detector.reset()
    assert detector.candidate_key is None
    assert detector.candidate_progress == 0.0

