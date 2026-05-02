import argparse
import json
import math
from pathlib import Path

import numpy as np

from .geometry import batch_pose_to_ego, cumulative_distance, perturb_pose, pose_to_ego
from .route_dataset import DEFAULT_HORIZONS, _parse_horizons


def _feature_vector(v, w, imu, image_valid, lidar_valid, current_pose, anchor_pose, lookahead_pose, progress, remaining_m):
    gx, gy, gyaw = pose_to_ego(*current_pose, *lookahead_pose)
    ax, ay, ayaw = pose_to_ego(*current_pose, *anchor_pose)
    return np.asarray([
        v,
        w,
        *imu.tolist(),
        float(image_valid),
        float(lidar_valid),
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


def build_sensor_dataset(args):
    rng = np.random.default_rng(args.seed)
    src = np.load(Path(args.input).expanduser(), allow_pickle=False)
    time = src["time"].astype(np.float64)
    x = src["x"].astype(np.float64)
    y = src["y"].astype(np.float64)
    yaw = src["yaw"].astype(np.float64)
    v = src["v"].astype(np.float64)
    w = src["w"].astype(np.float64)
    imu = src["imu"].astype(np.float32)
    images = src["images"]
    image_valid = src["image_valid"].astype(bool)
    lidar_bev = src["lidar_bev"]
    lidar_valid = src["lidar_valid"].astype(bool)
    source_meta = json.loads(src["meta"].item()) if "meta" in src.files else {}

    if args.require_exteroceptive:
        sensor_mask = image_valid | lidar_valid
    else:
        sensor_mask = np.ones(len(time), dtype=bool)

    route_s = cumulative_distance(x, y)
    route_len = float(route_s[-1]) if len(route_s) else 0.0
    horizons = np.asarray(args.horizons, dtype=np.float64)
    max_horizon = float(np.max(horizons))

    scalar_features = []
    targets = []
    base_indices = []
    sample_info = []

    for i in range(len(time)):
        if not sensor_mask[i]:
            continue
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
            perturbations.append((
                float(rng.uniform(-args.lateral_max_m, args.lateral_max_m)),
                float(rng.uniform(-args.forward_max_m, args.forward_max_m)),
                float(rng.uniform(-args.yaw_max_deg, args.yaw_max_deg) * math.pi / 180.0),
            ))

        for lateral, forward, yaw_delta in perturbations:
            current_pose = perturb_pose(anchor_pose[0], anchor_pose[1], anchor_pose[2], lateral, forward, yaw_delta)
            scalar_features.append(_feature_vector(
                float(v[i]),
                float(w[i]),
                imu[i],
                image_valid[i],
                lidar_valid[i],
                current_pose,
                anchor_pose,
                lookahead_pose,
                float(route_s[i] / max(route_len, 1e-6)),
                float(route_len - route_s[i]),
            ))
            targets.append(batch_pose_to_ego(np.asarray(current_pose), future_poses).reshape(-1).astype(np.float32))
            base_indices.append(i)
            sample_info.append((i, lookahead_idx, lateral, forward, yaw_delta))

    scalar_features = np.stack(scalar_features).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    sample_info = np.asarray(sample_info, dtype=np.float32)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": source_meta,
        "horizons": horizons.tolist(),
        "lookahead_m": args.lookahead_m,
        "augmentations": args.augmentations,
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
            "image_valid",
            "lidar_valid",
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
        scalar_features=scalar_features,
        targets=targets,
        base_indices=base_indices,
        sample_info=sample_info,
        images=images,
        image_valid=image_valid,
        lidar_bev=lidar_bev,
        lidar_valid=lidar_valid,
        route=np.stack([x, y, yaw, route_s], axis=1).astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(json.dumps({
        "output": str(out_path),
        "samples": int(len(targets)),
        "base_frames": int(len(time)),
        "feature_dim": int(scalar_features.shape[1]),
        "target_dim": int(targets.shape[1]),
        "image_coverage": float(np.mean(image_valid)),
        "lidar_coverage": float(np.mean(lidar_valid)),
        "route_length_m": route_len,
    }, ensure_ascii=False, indent=2))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build camera/LiDAR route-following dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", type=_parse_horizons, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--lookahead-m", type=float, default=2.0)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--lateral-max-m", type=float, default=1.0)
    parser.add_argument("--forward-max-m", type=float, default=0.5)
    parser.add_argument("--yaw-max-deg", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--require-exteroceptive", action="store_true")
    return parser


def main():
    build_sensor_dataset(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

