import numpy as np

from gazetype.vision import _pose_angles, _quadrilateral_eye_ratio


def test_pose_angles_extract_yaw_and_pitch() -> None:
    yaw = 0.25
    yaw_matrix = np.eye(4)
    yaw_matrix[0, 0] = np.cos(yaw)
    yaw_matrix[0, 2] = np.sin(yaw)
    yaw_matrix[2, 0] = -np.sin(yaw)
    yaw_matrix[2, 2] = np.cos(yaw)
    extracted_yaw, extracted_pitch = _pose_angles(yaw_matrix)
    assert np.isclose(extracted_yaw, yaw)
    assert np.isclose(extracted_pitch, 0.0)

    pitch = -0.2
    pitch_matrix = np.eye(4)
    pitch_matrix[1, 1] = np.cos(pitch)
    pitch_matrix[1, 2] = -np.sin(pitch)
    pitch_matrix[2, 1] = np.sin(pitch)
    pitch_matrix[2, 2] = np.cos(pitch)
    _, extracted_pitch = _pose_angles(pitch_matrix)
    assert np.isclose(extracted_pitch, pitch)


def test_quadrilateral_eye_ratio_is_stable_when_eye_is_tilted() -> None:
    angle = np.deg2rad(30.0)
    horizontal = np.asarray((np.cos(angle), np.sin(angle))) * 0.4
    vertical = np.asarray((-np.sin(angle), np.cos(angle))) * 0.1
    center = np.asarray((0.5, 0.5))
    expected = (0.75, 0.25)
    iris = center + horizontal * (expected[0] - 0.5) + vertical * (expected[1] - 0.5)
    actual = _quadrilateral_eye_ratio(
        tuple(iris),
        tuple(center - horizontal / 2),
        tuple(center + horizontal / 2),
        tuple(center - vertical / 2),
        tuple(center + vertical / 2),
    )
    assert np.allclose(actual, expected)
