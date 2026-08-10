import numpy as np
import scipy.io

from tools.datasets.mpiigaze import (
    angles_to_pseudo_target,
    gaze_vector_to_angles,
    iter_normalized_samples,
    summarize,
)


def _write_mat(root) -> None:
    normalized = root / "Data" / "Normalized" / "p00"
    original = root / "Data" / "Original" / "p00" / "day01"
    normalized.mkdir(parents=True)
    original.mkdir(parents=True)
    (original / "0001.jpg").write_bytes(b"fake")
    eye_dtype = np.dtype([("gaze", "O"), ("image", "O"), ("pose", "O")])
    data_dtype = np.dtype([("right", "O"), ("left", "O")])
    right = np.empty((1, 1), dtype=eye_dtype)
    left = np.empty((1, 1), dtype=eye_dtype)
    for eye in (right, left):
        eye["gaze"][0, 0] = np.asarray([[0.0, 0.0, -1.0]])
        eye["image"][0, 0] = np.zeros((1, 36, 60), dtype=np.uint8)
        eye["pose"][0, 0] = np.asarray([[0.1, 0.2, 0.3]])
    data = np.empty((1, 1), dtype=data_dtype)
    data["right"][0, 0] = right
    data["left"][0, 0] = left
    scipy.io.savemat(
        normalized / "day01.mat",
        {"data": data, "filenames": np.asarray([["0001.jpg"]], dtype=object)},
    )


def test_mpiigaze_normalized_mat_parser(tmp_path) -> None:
    _write_mat(tmp_path)
    samples = list(iter_normalized_samples(tmp_path))
    assert len(samples) == 2
    assert samples[0].participant == "p00"
    assert samples[0].day == "day01"
    assert samples[0].filename == "0001.jpg"
    assert samples[0].image_path.exists()
    assert samples[0].gaze_vector == (0.0, 0.0, -1.0)
    assert summarize(tmp_path)["sample_count"] == 2


def test_mpiigaze_gaze_vector_pseudo_target() -> None:
    yaw, pitch = gaze_vector_to_angles((0.0, 0.0, -1.0))
    assert yaw == 0.0
    assert pitch == 0.0
    assert angles_to_pseudo_target(yaw, pitch) == (0.5, 0.5)
