import json

from tools.build_mpiigaze_manifest import build_manifest_rows, infer_angle_ranges, write_manifest
from tools.datasets.mpiigaze import MpiiGazeSample


def _sample(image_path, gaze_vector=(0.0, 0.0, -1.0)) -> MpiiGazeSample:
    return MpiiGazeSample(
        participant="p00",
        day="day01",
        filename=image_path.name,
        eye="left",
        image_path=image_path,
        gaze_vector=gaze_vector,
        head_pose=(0.1, 0.2, 0.3),
    )


def test_manifest_rows_cache_features_and_write_jsonl(tmp_path) -> None:
    image_path = tmp_path / "0001.jpg"
    calls = []

    def extractor(path):
        calls.append(path)
        return (0.1,) * 10

    rows = list(
        build_manifest_rows(
            [_sample(image_path), _sample(image_path)],
            extractor,
            (-0.5, 0.5),
            (-0.35, 0.35),
        )
    )
    assert len(calls) == 1
    assert len(rows) == 2
    assert rows[0]["target_x"] == 0.5
    assert rows[0]["target_y"] == 0.5
    assert rows[0]["features"] == [0.1] * 10

    out = tmp_path / "manifest.jsonl"
    assert write_manifest(rows, out, progress_every=0) == 2
    written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert written[0]["dataset"] == "mpiigaze"


def test_manifest_builder_skips_images_without_detected_features(tmp_path) -> None:
    rows = list(
        build_manifest_rows(
            [_sample(tmp_path / "missing.jpg")],
            lambda _path: None,
            (-0.5, 0.5),
            (-0.35, 0.35),
        )
    )
    assert rows == []


def test_infer_angle_ranges_uses_percentiles(tmp_path) -> None:
    yaw_range, pitch_range = infer_angle_ranges([
        _sample(tmp_path / "a.jpg", (-0.5, 0.0, -1.0)),
        _sample(tmp_path / "b.jpg", (0.0, 0.0, -1.0)),
        _sample(tmp_path / "c.jpg", (0.5, 0.2, -1.0)),
    ])
    assert yaw_range[0] < 0.0 < yaw_range[1]
    assert pitch_range[0] <= 0.0 < pitch_range[1]
