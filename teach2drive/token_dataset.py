import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .geometry import batch_pose_to_ego, cumulative_distance, perturb_pose, pose_to_ego, wrap_angle
from .route_dataset import DEFAULT_HORIZONS, _parse_horizons


FEATURE_NAMES = [
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
]

STOP_STATE_NAMES = ["drive", "approach_stop", "stopped_waiting", "release_go"]
STOP_REASON_NAMES = [
    "none",
    "unknown_stop",
    "startup",
    "route_end",
    "traffic_light",
    "stop_sign",
    "front_vehicle",
    "junction_yield",
]


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frames(path: Path) -> List[Dict]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    frames.sort(key=lambda item: (int(item.get("step", 0)), float(item.get("time", 0.0))))
    return frames


def _has_required_tokens(episode_dir: Path, frame: Dict, cameras: Sequence[str]) -> bool:
    camera_tokens = frame.get("camera_tokens", {})
    for camera in cameras:
        token = camera_tokens.get(camera)
        if not token or not (episode_dir / token).exists():
            return False
    lidar_token = frame.get("lidar_bev_token")
    return bool(lidar_token and (episode_dir / lidar_token).exists())


def _odom_arrays(frames: Sequence[Dict]):
    time = np.asarray([float(frame["time"]) for frame in frames], dtype=np.float64)
    x = np.asarray([float(frame["odom"]["x"]) for frame in frames], dtype=np.float64)
    y = np.asarray([float(frame["odom"]["y"]) for frame in frames], dtype=np.float64)
    yaw = np.asarray([float(frame["odom"]["yaw"]) for frame in frames], dtype=np.float64)
    v = np.asarray([float(frame["odom"].get("v_forward", 0.0)) for frame in frames], dtype=np.float64)
    w = np.asarray([float(frame["odom"].get("yaw_rate", 0.0)) for frame in frames], dtype=np.float64)
    imu = np.asarray([
        [
            *frame.get("imu", {}).get("accelerometer", [0.0, 0.0, 0.0]),
            *frame.get("imu", {}).get("gyroscope", [0.0, 0.0, 0.0]),
        ]
        for frame in frames
    ], dtype=np.float32)
    return time, x, y, yaw, v, w, imu


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
        1.0,
        1.0,
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


def _control_label(frame: Dict):
    control = frame.get("control")
    if not control:
        return np.zeros(3, dtype=np.float32), 0.0
    return np.asarray([
        float(control.get("steer", 0.0)),
        float(control.get("throttle", 0.0)),
        float(control.get("brake", 0.0)),
    ], dtype=np.float32), 1.0


def _lane_label(frame: Dict, lateral_offset: float = 0.0, yaw_delta: float = 0.0):
    lane = frame.get("lane") or {}
    if not lane.get("valid", False):
        return np.zeros(2, dtype=np.float32), 0.0
    center_offset = float(lane.get("lane_center_offset_m", 0.0)) + float(lateral_offset)
    heading_error = float(wrap_angle(float(lane.get("lane_heading_error_rad", 0.0)) + float(yaw_delta)))
    return np.asarray([center_offset, heading_error], dtype=np.float32), 1.0


def _phase_weight(phase: str, args: argparse.Namespace) -> float:
    if phase == "stopped_start":
        return float(args.stopped_start_weight)
    if phase == "stopped_end":
        return float(args.stopped_end_weight)
    return float(args.drive_weight)


def _stop_state_label(current_speed: float, future_speeds: np.ndarray, args: argparse.Namespace) -> int:
    current_speed = abs(float(current_speed))
    future_abs = np.abs(future_speeds.astype(np.float32))
    future_min = float(np.min(future_abs))
    future_max = float(np.max(future_abs))
    if current_speed <= args.stop_state_stop_speed:
        if future_max >= args.stop_state_move_speed:
            return STOP_STATE_NAMES.index("release_go")
        return STOP_STATE_NAMES.index("stopped_waiting")
    if future_min <= args.stop_state_stop_speed:
        return STOP_STATE_NAMES.index("approach_stop")
    return STOP_STATE_NAMES.index("drive")


def _stop_reason_label(frame: Dict, phase: str, remaining_m: float, stop_state: int, args: argparse.Namespace):
    state_name = STOP_STATE_NAMES[int(stop_state)]
    if state_name == "drive":
        return STOP_REASON_NAMES.index("none"), 1.0
    if phase == "stopped_start":
        return STOP_REASON_NAMES.index("startup"), 1.0
    if phase == "stopped_end" or remaining_m <= args.route_end_reason_m:
        return STOP_REASON_NAMES.index("route_end"), 1.0

    traffic_light = frame.get("traffic_light") or {}
    light_state = str(traffic_light.get("state", "")).lower()
    if traffic_light.get("is_at_traffic_light") and light_state in {"red", "yellow"}:
        return STOP_REASON_NAMES.index("traffic_light"), 1.0

    stop_sign = frame.get("stop_sign") or {}
    if stop_sign.get("valid") and stop_sign.get("distance_m") is not None:
        if float(stop_sign.get("distance_m")) <= args.stop_sign_reason_m:
            return STOP_REASON_NAMES.index("stop_sign"), 1.0

    front_vehicle = frame.get("front_vehicle") or {}
    if front_vehicle.get("valid") and front_vehicle.get("distance_m") is not None:
        if float(front_vehicle.get("distance_m")) <= args.front_vehicle_reason_m:
            return STOP_REASON_NAMES.index("front_vehicle"), 1.0

    lane = frame.get("lane") or {}
    if lane.get("valid") and lane.get("is_junction"):
        return STOP_REASON_NAMES.index("junction_yield"), 1.0

    return STOP_REASON_NAMES.index("unknown_stop"), float(args.train_unknown_stop_reason)


def _discover_episode_dirs(input_roots: Sequence[str]) -> List[Path]:
    episode_dirs = []
    for root_text in input_roots:
        root = Path(root_text).expanduser()
        if (root / "frames.jsonl").exists():
            episode_dirs.append(root.resolve())
            continue
        episode_dirs.extend(sorted(path.resolve() for path in root.glob("episode_*") if (path / "frames.jsonl").exists()))
    return sorted(dict.fromkeys(episode_dirs))


def build_token_dataset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    cameras = list(args.cameras)
    horizons = np.asarray(args.horizons, dtype=np.float64)
    max_horizon = float(np.max(horizons))
    episode_dirs = _discover_episode_dirs(args.input_root)
    if not episode_dirs:
        raise RuntimeError(f"No tokenized episodes found under: {args.input_root}")

    scalar_features = []
    traj_targets = []
    speed_targets = []
    stop_targets = []
    stop_state_targets = []
    stop_reason_targets = []
    stop_reason_masks = []
    control_targets = []
    control_masks = []
    lane_targets = []
    lane_masks = []
    sample_weights = []
    sample_episode_indices = []
    sample_frame_indices = []
    sample_info = []
    episode_summaries = []
    total_frames = 0
    valid_frames = 0

    for episode_idx, episode_dir in enumerate(episode_dirs):
        raw_frames = _read_frames(episode_dir / "frames.jsonl")
        total_frames += len(raw_frames)
        keep_indices = [idx for idx, frame in enumerate(raw_frames) if _has_required_tokens(episode_dir, frame, cameras)]
        if len(keep_indices) < 2:
            continue
        frames = [raw_frames[idx] for idx in keep_indices]
        valid_frames += len(frames)
        time, x, y, yaw, v, w, imu = _odom_arrays(frames)
        if len(time) < 2 or time[-1] < max_horizon:
            continue

        route_s = cumulative_distance(x, y)
        route_len = float(route_s[-1]) if len(route_s) else 0.0
        episode_samples = 0

        for frame_idx in range(len(frames)):
            if time[frame_idx] + max_horizon > time[-1]:
                continue
            future_indices = np.searchsorted(time, time[frame_idx] + horizons)
            if np.any(future_indices >= len(time)):
                continue

            lookahead_idx = int(np.searchsorted(route_s, route_s[frame_idx] + args.lookahead_m))
            lookahead_idx = min(lookahead_idx, len(time) - 1)
            anchor_pose = (float(x[frame_idx]), float(y[frame_idx]), float(yaw[frame_idx]))
            lookahead_pose = (float(x[lookahead_idx]), float(y[lookahead_idx]), float(yaw[lookahead_idx]))
            future_poses = np.stack([x[future_indices], y[future_indices], yaw[future_indices]], axis=1)
            future_speeds = v[future_indices].astype(np.float32)

            perturbations = [(0.0, 0.0, 0.0)]
            for _ in range(args.augmentations):
                perturbations.append((
                    float(rng.uniform(-args.lateral_max_m, args.lateral_max_m)),
                    float(rng.uniform(-args.forward_max_m, args.forward_max_m)),
                    float(rng.uniform(-args.yaw_max_deg, args.yaw_max_deg) * math.pi / 180.0),
                ))

            stop_target = float(np.max(np.abs(future_speeds)) <= args.stop_speed_threshold)
            control_target, control_mask = _control_label(frames[frame_idx])
            phase = str(frames[frame_idx].get("phase", "drive"))
            remaining_m = float(route_len - route_s[frame_idx])
            stop_state_target = _stop_state_label(float(v[frame_idx]), future_speeds, args)
            stop_reason_target, stop_reason_mask = _stop_reason_label(frames[frame_idx], phase, remaining_m, stop_state_target, args)
            sample_weight = _phase_weight(phase, args)
            for lateral, forward, yaw_delta in perturbations:
                current_pose = perturb_pose(anchor_pose[0], anchor_pose[1], anchor_pose[2], lateral, forward, yaw_delta)
                lane_target, lane_mask = _lane_label(frames[frame_idx], lateral, yaw_delta)
                scalar_features.append(_feature_vector(
                    float(v[frame_idx]),
                    float(w[frame_idx]),
                    imu[frame_idx],
                    current_pose,
                    anchor_pose,
                    lookahead_pose,
                    float(route_s[frame_idx] / max(route_len, 1e-6)),
                    float(route_len - route_s[frame_idx]),
                ))
                traj_targets.append(batch_pose_to_ego(np.asarray(current_pose, dtype=np.float64), future_poses).reshape(-1).astype(np.float32))
                speed_targets.append(future_speeds.astype(np.float32))
                stop_targets.append(stop_target)
                stop_state_targets.append(stop_state_target)
                stop_reason_targets.append(stop_reason_target)
                stop_reason_masks.append(stop_reason_mask)
                control_targets.append(control_target)
                control_masks.append(control_mask)
                lane_targets.append(lane_target)
                lane_masks.append(lane_mask)
                sample_weights.append(sample_weight)
                sample_episode_indices.append(episode_idx)
                raw_frame_idx = keep_indices[frame_idx]
                sample_frame_indices.append(raw_frame_idx)
                sample_info.append((episode_idx, raw_frame_idx, lookahead_idx, lateral, forward, yaw_delta))
                episode_samples += 1

        episode_summaries.append({
            "episode_dir": str(episode_dir),
            "frames": len(frames),
            "route_length_m": route_len,
            "samples": episode_samples,
            "meta": _read_json(episode_dir / "episode_meta.json"),
        })

    if not scalar_features:
        raise RuntimeError("No training samples were generated. Check episode length, tokens, and horizons.")

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "teach2drive_tokenized_carla",
        "input_roots": [str(Path(path).expanduser()) for path in args.input_root],
        "episode_count": len(episode_dirs),
        "episodes": episode_summaries,
        "cameras": cameras,
        "horizons": horizons.tolist(),
        "lookahead_m": args.lookahead_m,
        "augmentations": args.augmentations,
        "lateral_max_m": args.lateral_max_m,
        "forward_max_m": args.forward_max_m,
        "yaw_max_deg": args.yaw_max_deg,
        "stop_speed_threshold": args.stop_speed_threshold,
        "drive_weight": args.drive_weight,
        "stopped_start_weight": args.stopped_start_weight,
        "stopped_end_weight": args.stopped_end_weight,
        "feature_names": FEATURE_NAMES,
        "traj_target": "flattened [dx, dy, dyaw] for each horizon in ego frame",
        "speed_target": "future v_forward for each horizon",
        "stop_target": "1 when all future horizon speeds are below threshold",
        "control_target": "optional [steer, throttle, brake] from vehicle.get_control(); mask=0 when unavailable",
        "lane_target": "optional [lane_center_offset_m, lane_heading_error_rad]; mask=0 when unavailable",
        "stop_state_names": STOP_STATE_NAMES,
        "stop_reason_names": STOP_REASON_NAMES,
        "stop_state_target": "classification: drive, approach_stop, stopped_waiting, release_go",
        "stop_reason_target": "classification with mask: none, unknown_stop, startup, route_end, traffic_light, stop_sign, front_vehicle, junction_yield",
    }

    np.savez_compressed(
        out_path,
        scalar_features=np.stack(scalar_features).astype(np.float32),
        traj_targets=np.stack(traj_targets).astype(np.float32),
        speed_targets=np.stack(speed_targets).astype(np.float32),
        stop_targets=np.asarray(stop_targets, dtype=np.float32),
        stop_state_targets=np.asarray(stop_state_targets, dtype=np.int64),
        stop_reason_targets=np.asarray(stop_reason_targets, dtype=np.int64),
        stop_reason_masks=np.asarray(stop_reason_masks, dtype=np.float32),
        control_targets=np.stack(control_targets).astype(np.float32),
        control_masks=np.asarray(control_masks, dtype=np.float32),
        lane_targets=np.stack(lane_targets).astype(np.float32),
        lane_masks=np.asarray(lane_masks, dtype=np.float32),
        sample_weights=np.asarray(sample_weights, dtype=np.float32),
        sample_episode_indices=np.asarray(sample_episode_indices, dtype=np.int64),
        sample_frame_indices=np.asarray(sample_frame_indices, dtype=np.int64),
        sample_info=np.asarray(sample_info, dtype=np.float32),
        episode_dirs=np.asarray([str(path) for path in episode_dirs]),
        cameras=np.asarray(cameras),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(json.dumps({
        "output": str(out_path),
        "episodes": int(len(episode_dirs)),
        "total_frames": int(total_frames),
        "valid_frames": int(valid_frames),
        "samples": int(len(scalar_features)),
        "feature_dim": int(len(FEATURE_NAMES)),
        "traj_dim": int(len(traj_targets[0])),
        "speed_dim": int(len(speed_targets[0])),
        "stop_positive_ratio": float(np.mean(stop_targets)),
        "stop_state_distribution": {
            name: float(np.mean(np.asarray(stop_state_targets) == idx))
            for idx, name in enumerate(STOP_STATE_NAMES)
        },
        "stop_reason_label_ratio": float(np.mean(stop_reason_masks)),
        "stop_reason_distribution": {
            name: float(np.mean(np.asarray(stop_reason_targets) == idx))
            for idx, name in enumerate(STOP_REASON_NAMES)
        },
        "control_label_ratio": float(np.mean(control_masks)),
        "lane_label_ratio": float(np.mean(lane_masks)),
        "mean_sample_weight": float(np.mean(sample_weights)),
        "cameras": cameras,
        "horizons": horizons.tolist(),
    }, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a lightweight index for tokenized CARLA camera/LiDAR data.")
    parser.add_argument("--input-root", nargs="+", required=True, help="Tokenized dataset root(s) or episode directory.")
    parser.add_argument("--output", required=True, help="Output token index .npz.")
    parser.add_argument("--cameras", "--camera-names", dest="camera_names", default="front,left,right", help="Comma separated camera token names.")
    parser.add_argument("--horizons", type=_parse_horizons, default=list(DEFAULT_HORIZONS), help="Comma separated future horizons in seconds.")
    parser.add_argument("--lookahead-m", type=float, default=8.0)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--lateral-max-m", type=float, default=1.2)
    parser.add_argument("--forward-max-m", type=float, default=0.7)
    parser.add_argument("--yaw-max-deg", type=float, default=45.0)
    parser.add_argument("--stop-speed-threshold", type=float, default=0.2)
    parser.add_argument("--stop-state-stop-speed", type=float, default=0.35)
    parser.add_argument("--stop-state-move-speed", type=float, default=1.0)
    parser.add_argument("--route-end-reason-m", type=float, default=8.0)
    parser.add_argument("--stop-sign-reason-m", type=float, default=18.0)
    parser.add_argument("--front-vehicle-reason-m", type=float, default=12.0)
    parser.add_argument("--train-unknown-stop-reason", action="store_true")
    parser.add_argument("--drive-weight", type=float, default=1.0)
    parser.add_argument("--stopped-start-weight", type=float, default=0.25)
    parser.add_argument("--stopped-end-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=31)
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.cameras = [item.strip() for item in args.camera_names.split(",") if item.strip()]
    if not args.cameras:
        raise ValueError("At least one camera name is required.")
    return args


def main() -> None:
    build_token_dataset(_normalize_args(build_arg_parser().parse_args()))


if __name__ == "__main__":
    main()
