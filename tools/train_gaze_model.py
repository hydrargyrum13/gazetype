from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from tools.datasets import weyeds
except ModuleNotFoundError:
    from datasets import weyeds


def _design(features: np.ndarray, polynomial_degree: int) -> np.ndarray:
    columns = [np.ones(len(features))]
    columns.extend(features[:, index] for index in range(features.shape[1]))
    if polynomial_degree >= 2:
        for first in range(features.shape[1]):
            for second in range(first, features.shape[1]):
                columns.append(features[:, first] * features[:, second])
    return np.column_stack(columns)


def _quick_samples() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    features = rng.normal(0.0, 0.25, size=(80, 10))
    targets = np.column_stack((
        np.clip(0.5 + features[:, 0] * 0.35 + features[:, 4] * 0.08, 0.0, 1.0),
        np.clip(0.5 + features[:, 1] * 0.40 + features[:, 5] * 0.08, 0.0, 1.0),
    ))
    return features, targets


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    polynomial_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[1] != 10:
        raise ValueError("Training requires 10 gaze features per sample")
    if targets.shape != (len(features), 2):
        raise ValueError("Training targets must be paired x/y values")
    mean = np.mean(features, axis=0)
    std = np.maximum(np.std(features, axis=0), 1e-6)
    design = _design((features - mean) / std, polynomial_degree)
    regularizer = ridge * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    return mean, std, weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local Gazetype general gaze model.")
    parser.add_argument("--dataset", choices=("weyeds",), default="weyeds")
    parser.add_argument("--data-dir", type=Path, help="Local dataset directory with a normalized manifest.")
    parser.add_argument("--out", type=Path, default=Path("models/gazetype_general.npz"))
    parser.add_argument("--quick", action="store_true", help="Run a small synthetic smoke training job.")
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--polynomial-degree", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        features, targets = _quick_samples()
    else:
        if args.data_dir is None:
            raise SystemExit("--data-dir is required unless --quick is used")
        if args.dataset == "weyeds":
            sample_features, sample_targets = weyeds.arrays(weyeds.load_dataset(args.data_dir))
        else:
            raise SystemExit(f"Unsupported dataset: {args.dataset}")
        features = np.asarray(sample_features, dtype=np.float64)
        targets = np.asarray(sample_targets, dtype=np.float64)

    mean, std, weights = fit_ridge(features, targets, args.ridge, args.polynomial_degree)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        mean=mean,
        std=std,
        weights=weights,
        polynomial_degree=np.asarray(args.polynomial_degree, dtype=np.int64),
    )
    print(f"Wrote {args.out} with {len(features)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
