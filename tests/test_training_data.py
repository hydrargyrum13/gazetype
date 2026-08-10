import json

from gazetype.training_data import TrainingSampleWriter


def test_jsonl_sample_writer_records_calibration_sample(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    writer = TrainingSampleWriter(path, "10,20,1920,1080", 2)
    writer.write((0.25, 0.75), (0.1,) * 10, {"selected_sample_count": 8})

    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["target_x"] == 0.25
    assert row["target_y"] == 0.75
    assert row["features"] == [0.1] * 10
    assert row["screen_geometry"] == {"x": 10, "y": 20, "width": 1920, "height": 1080}
    assert row["camera_index"] == 2
    assert row["quality"]["selected_sample_count"] == 8
