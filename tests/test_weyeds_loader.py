import json

from tools.datasets.weyeds import load_dataset, load_normalized_manifest


def test_weyeds_normalized_jsonl_manifest_parser(tmp_path) -> None:
    manifest = tmp_path / "gazetype_manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "image_path": "frame.png",
            "target_x": 0.2,
            "target_y": 0.8,
            "features": [0.1] * 10,
            "screen_width": 1920,
            "screen_height": 1080,
        })
        + "\n",
        encoding="utf-8",
    )

    sample = load_dataset(tmp_path)[0]
    assert sample.image_path == "frame.png"
    assert sample.target == (0.2, 0.8)
    assert sample.features == (0.1,) * 10
    assert sample.screen_width == 1920


def test_weyeds_normalized_csv_manifest_parser(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "target_x,target_y,feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9\n"
        "0.4,0.6,0,1,2,3,4,5,6,7,8,9\n",
        encoding="utf-8",
    )

    sample = load_normalized_manifest(manifest)[0]
    assert sample.target == (0.4, 0.6)
    assert sample.features == tuple(float(value) for value in range(10))


def test_weyeds_loader_accepts_training_samples_jsonl(tmp_path) -> None:
    manifest = tmp_path / "training_samples.jsonl"
    manifest.write_text(
        json.dumps({
            "target_x": 0.3,
            "target_y": 0.7,
            "features": [0.2] * 10,
        })
        + "\n",
        encoding="utf-8",
    )

    sample = load_dataset(tmp_path)[0]
    assert sample.target == (0.3, 0.7)
    assert sample.features == (0.2,) * 10
