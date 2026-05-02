import argparse
import json
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .geometry import batch_pose_to_ego, cumulative_distance, perturb_pose, pose_to_ego, wrap_angle


DEFAULT_HORIZONS = (0.5, 1.0, 1.5, 2.0)


def _parse_horizons(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _feature_vector(
    v: float,
    w: float,
    imu: np.ndarray,
    current_pose: Sequence[float],
    anchor_pose: Sequence[float],
    lookahead_pose: Sequence[float],
    progress: float,
    remaining_m: float,
) -> np.ndarray:
    gx, gy, gyaw = pose_to_ego(*current_pose, *lookahead_pose)
    ax, ay, ayaw = pose_to_ego(*current_pose, *anchor_pose)
    return np.asarray([
        v,
        w,
        *imu.tolist(),
        gx,
        gy,
        math.sin(gyaw),
        math.cos(gyaw),
        ax,
        ay,
        math.sin(ayaw),
        math.cos(ayaw),
        progress,
        remaining_m,
    ], dtype=np.float32)


def build_dataset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    src = np.load(Path(args.input).expanduser(), allow_pickle=False)
    time = src["time"].astype(np.float64)
    x = src["x"].astype(np.float64)
    y = src["y"].astype(np.float64)
    yaw = src["yaw"].astype(np.float64)
    v = src["v"].astype(np.float64)
    w = src["w"].astype(np.float64)
    imu = src["imu"].astype(np.float32) if "imu" in src.files else np.zeros((len(time), 6), dtype=np.float32)
    source_meta = json.loads(src["meta"].item()) if "meta" in src.files else {}

    route_s = cumulative_distance(x, y)
    route_len = float(route_s[-1]) if len(route_s) else 0.0
    horizons = np.asarray(args.horizons, dtype=np.float64)

    features = []
    targets = []
    sample_info = []
    max_horizon = float(np.max(horizons))

    for i in range(len(time)):
        if time[i] + max_horizon > time[-1]:
            continue
        future_indices = np.searchsorted(time, time[i] + horizons)
        if np.any(future_indices >= len(time)):
            continue
        lookahead_idx = int(np.searchsorted(route_s, route_s[i] + args.lookahead_m))
        lookahead_idx = min(lookahead_idx, len(time) - 1)

        anchor_pose = (float(x[i]), float(y[i]), float(yaw[i]))
        lookahead_pose = (float(x[lookahead_idx]), float(y[lookahead_idx]), float(yaw[lookahead_idx]))
        future_poses = np.stack([x[future_indices], y[future_indices], yaw[future_indices]], axis=1)

        perturbations = [(0.0, 0.0, 0.0)]
        for _ in range(args.augmentations):
            lateral = float(rng.uniform(-args.lateral_max_m, args.lateral_max_m))
            forward = float(rng.uniform(-args.forward_max_m, args.forward_max_m))
            yaw_delta = float(rng.uniform(-args.yaw_max_deg, args.yaw_max_deg) * math.pi / 180.0)
            perturbations.append((lateral, forward, yaw_delta))

        for lateral, forward, yaw_delta in perturbations:
            current_pose = perturb_pose(anchor_pose[0], anchor_pose[1], anchor_pose[2], lateral, forward, yaw_delta)
            feat = _feature_vector(
                float(v[i]),
                float(w[i]),
                imu[i],
                current_pose,
                anchor_pose,
                lookahead_pose,
                float(route_s[i] / max(route_len, 1e-6)),
                float(route_len - route_s[i]),
            )
            target = batch_pose_to_ego(np.asarray(current_pose, dtype=np.float64), future_poses).reshape(-1).astype(np.float32)
            features.append(feat)
            targets.append(target)
            sample_info.append((i, lookahead_idx, lateral, forward, yaw_delta))

    if not features:
        raise RuntimeError("No training samples were generated. Check bag duration and horizons.")

    features_np = np.stack(features).astype(np.float32)
    targets_np = np.stack(targets).astype(np.float32)
    sample_info_np = np.asarray(sample_info, dtype=np.float32)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "source": source_meta,
        "horizons": horizons.tolist(),
        "lookahead_m": args.lookahead_m,
        "augmentations": args.augmentations,
        "lateral_max_m": args.lateral_max_m,
        "forward_max_m": args.forward_max_m,
        "yaw_max_deg": args.yaw_max_deg,
        "route_length_m": route_len,
        "feature_names": [
            "v",
            "w",
            "imu_ax",
            "imu_ay",
            "imu_az",
            "imu_gx",
            "imu_gy",
            "imu_gz",
            "lookahead_x_ego",
            "lookahead_y_ego",
            "sin_lookahead_yaw",
            "cos_lookahead_yaw",
            "anchor_x_ego",
            "anchor_y_ego",
            "sin_anchor_yaw",
            "cos_anchor_yaw",
            "route_progress",
            "remaining_m",
        ],
        "target": "flattened [dx, dy, dyaw] for each horizon in ego frame",
    }

    np.savez_compressed(
        out_path,
        features=features_np,
        targets=targets_np,
        sample_info=sample_info_np,
        time=time.astype(np.float32),
        route=np.stack([x, y, yaw, route_s], axis=1).astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(json.dumps({
        "output": str(out_path),
        "samples": int(len(features_np)),
        "feature_dim": int(features_np.shape[1]),
        "target_dim": int(targets_np.shape[1]),
        "route_length_m": route_len,
        "horizons": horizons.tolist(),
    }, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build route-following supervised samples.")
    parser.add_argument("--input", required=True, help="Extracted route .npz from rosbag_extract.")
    parser.add_argument("--output", required=True, help="Output train dataset .npz.")
    parser.add_argument("--horizons", type=_parse_horizons, default=list(DEFAULT_HORIZONS), help="Comma separated future horizons in seconds.")
    parser.add_argument("--lookahead-m", type=float, default=2.0, help="Route lookahead distance used as local goal.")
    parser.add_argument("--augmentations", type=int, default=4, help="Pose perturbations per real odom sample.")
    parser.add_argument("--lateral-max-m", type=float, default=1.0, help="Max lateral perturbation for route rejoin training.")
    parser.add_argument("--forward-max-m", type=float, default=0.5, help="Max forward/backward perturbation.")
    parser.add_argument("--yaw-max-deg", type=float, default=60.0, help="Max heading perturbation in degrees.")
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    build_dataset(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
