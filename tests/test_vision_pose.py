import numpy as np

from gazetype.vision import _pose_angles


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
