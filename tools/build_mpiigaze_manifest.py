from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from gazetype.runtime import configure_numeric_locale
from gazetype.vision import extract_gaze_features, model_path
from tools.datasets.mpiigaze import (
    MpiiGazeSample,
    angles_to_pseudo_target,
    gaze_vector_to_angles,
    iter_normalized_samples,
)


FeatureExtractor = Callable[[Path], tuple[float, ...] | None]


def infer_angle_ranges(samples: Iterable[MpiiGazeSample]) -> tuple[tuple[float, float], tuple[float, float]]:
    yaws = []
    pitches = []
    for sample in samples:
        yaw, pitch = gaze_vector_to_angles(sample.gaze_vector)
        yaws.append(yaw)
        pitches.append(pitch)
    if not yaws:
        raise ValueError("No MPIIGaze samples found")
    yaw_range = (float(np.percentile(yaws, 1)), float(np.percentile(yaws, 99)))
    pitch_range = (float(np.percentile(pitches, 1)), float(np.percentile(pitches, 99)))
    return yaw_range, pitch_range


class MediaPipeFeatureExtractor:
    def __init__(self) -> None:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        configure_numeric_locale()
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp = mp
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path())),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self._detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def __call__(self, image_path: Path) -> tuple[float, ...] | None:
        image = self._cv2.imread(str(image_path))
        if image is None:
            return None
        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        transformation = (
            result.facial_transformation_matrixes[0]
            if result.facial_transformation_matrixes
            else None
        )
        return extract_gaze_features(result.face_landmarks[0], transformation)


def build_manifest_rows(
    samples: Iterable[MpiiGazeSample],
    extract_features: FeatureExtractor,
    yaw_range: tuple[float, float],
    pitch_range: tuple[float, float],
) -> Iterable[dict[str, object]]:
    feature_cache: dict[Path, tuple[float, ...] | None] = {}
    for sample in samples:
        if sample.image_path not in feature_cache:
            feature_cache[sample.image_path] = extract_features(sample.image_path)
        features = feature_cache[sample.image_path]
        if features is None:
            continue
        yaw, pitch = gaze_vector_to_angles(sample.gaze_vector)
        target_x, target_y = angles_to_pseudo_target(yaw, pitch, yaw_range, pitch_range)
        yield {
            "image_path": str(sample.image_path),
            "target_x": target_x,
            "target_y": target_y,
            "features": list(features),
            "dataset": "mpiigaze",
            "participant": sample.participant,
            "day": sample.day,
            "filename": sample.filename,
            "eye": sample.eye,
            "gaze_yaw": yaw,
            "gaze_pitch": pitch,
            "gaze_vector": list(sample.gaze_vector),
            "head_pose": list(sample.head_pose),
        }


def write_manifest(
    rows: Iterable[dict[str, object]],
    out_path: Path,
    progress_every: int = 100,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"wrote {count} rows")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Gazetype normalized JSONL manifest from MPIIGaze raw images."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/mpiigaze_gazetype_manifest.jsonl"))
    parser.add_argument("--limit", type=int, help="Limit MPIIGaze eye samples for smoke runs.")
    parser.add_argument("--eye", choices=("both", "left", "right"), default="both")
    parser.add_argument("--auto-range", action="store_true", help="Infer pseudo-target angle ranges.")
    parser.add_argument("--yaw-min", type=float, default=-0.5)
    parser.add_argument("--yaw-max", type=float, default=0.5)
    parser.add_argument("--pitch-min", type=float, default=-0.35)
    parser.add_argument("--pitch-max", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def _selected_samples(args: argparse.Namespace) -> list[MpiiGazeSample]:
    samples = list(iter_normalized_samples(args.data_dir, limit=args.limit))
    if args.eye == "both":
        return samples
    return [sample for sample in samples if sample.eye == args.eye]


def main() -> int:
    args = parse_args()
    samples = _selected_samples(args)
    if args.auto_range:
        yaw_range, pitch_range = infer_angle_ranges(samples)
    else:
        yaw_range = (args.yaw_min, args.yaw_max)
        pitch_range = (args.pitch_min, args.pitch_max)
    print({
        "input_samples": len(samples),
        "yaw_range": yaw_range,
        "pitch_range": pitch_range,
        "out": str(args.out),
    }, flush=True)
    print("initializing mediapipe face landmarker", flush=True)
    extractor = MediaPipeFeatureExtractor()
    try:
        print("extracting gazetype features", flush=True)
        rows = build_manifest_rows(samples, extractor, yaw_range, pitch_range)
        written = write_manifest(rows, args.out, progress_every=args.progress_every)
    finally:
        extractor.close()
    print({"written_rows": written}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
