import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import pseudo_label as base
from .token_dataset import STOP_REASON_NAMES, STOP_STATE_NAMES


def _parse_csv(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one camera name.")
    return items


def _tag_detection(candidate: Dict, camera_name: str) -> Dict:
    tagged = dict(candidate)
    tagged["camera"] = camera_name
    return tagged


def _tag_camera_label(label: Dict, camera_name: str) -> Dict:
    return {
        "camera": camera_name,
        "stop_sign": _tag_detection(label.get("stop_sign", {}), camera_name),
        "traffic_light": _tag_detection(label.get("traffic_light", {}), camera_name),
        "front_camera_vehicle": _tag_detection(label.get("front_camera_vehicle", {}), camera_name),
    }


def _empty_camera_label(camera_name: str) -> Dict:
    return _tag_camera_label(base._empty_camera_labels(), camera_name)


def _frame_camera_path(episode_dir: Path, frame: Dict, camera_name: str) -> Optional[Path]:
    token = frame.get("camera_tokens", {}).get(camera_name)
    if not token:
        return None
    return episode_dir / token


def _flatten_camera_paths(episode_dir: Path, frames: Sequence[Dict], cameras: Sequence[str]):
    frame_indices: List[int] = []
    camera_names: List[str] = []
    image_paths: List[Optional[Path]] = []
    for frame_idx, frame in enumerate(frames):
        for camera_name in cameras:
            frame_indices.append(frame_idx)
            camera_names.append(camera_name)
            image_paths.append(_frame_camera_path(episode_dir, frame, camera_name))
    return frame_indices, camera_names, image_paths


def _yolo_multicam_labels(model, image_paths: Sequence[Optional[Path]], camera_names: Sequence[str], args: argparse.Namespace) -> List[Dict]:
    labels = [_empty_camera_label(camera_name) for camera_name in camera_names]
    if model is None:
        return labels

    names = getattr(model, "names", {})
    for chunk_start in range(0, len(image_paths), args.yolo_chunk):
        chunk_paths = image_paths[chunk_start : chunk_start + args.yolo_chunk]
        chunk_cameras = camera_names[chunk_start : chunk_start + args.yolo_chunk]
        valid_pairs = [
            (offset, path, camera_name)
            for offset, (path, camera_name) in enumerate(zip(chunk_paths, chunk_cameras))
            if path is not None and path.exists()
        ]
        if not valid_pairs:
            continue

        results = model.predict(
            source=[str(path) for _offset, path, _camera_name in valid_pairs],
            stream=False,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_confidence,
            iou=args.yolo_iou,
            device=args.yolo_device or None,
            batch=args.yolo_batch,
            verbose=False,
        )
        for (offset, image_path, camera_name), result in zip(valid_pairs, results):
            label = base._yolo_result_to_label(result, image_path, names, args)
            labels[chunk_start + offset] = _tag_camera_label(label, camera_name)
    return labels


def _color_multicam_labels(image_paths: Sequence[Optional[Path]], camera_names: Sequence[str], args: argparse.Namespace) -> List[Dict]:
    labels = []
    for image_path, camera_name in zip(image_paths, camera_names):
        if image_path is not None and image_path.exists():
            label = {
                "stop_sign": base._detect_stop_sign(image_path, args),
                "traffic_light": base._detect_traffic_light(image_path, args),
                "front_camera_vehicle": base._empty_camera_labels()["front_camera_vehicle"],
            }
            labels.append(_tag_camera_label(label, camera_name))
        else:
            labels.append(_empty_camera_label(camera_name))
    return labels


def _group_labels_by_frame(frame_indices: Sequence[int], labels: Sequence[Dict], frame_count: int) -> List[List[Dict]]:
    grouped = [[] for _ in range(frame_count)]
    for frame_idx, label in zip(frame_indices, labels):
        grouped[frame_idx].append(label)
    return grouped


def _flatten_labels(label_groups: Sequence[Sequence[Dict]]) -> List[Dict]:
    return [label for group in label_groups for label in group]


def _label_episode(episode_dir: Path, args: argparse.Namespace) -> Dict:
    frames = base._read_frames(episode_dir / "frames.jsonl")
    times = np.asarray([float(frame.get("time", idx / max(args.hz, 1e-6))) for idx, frame in enumerate(frames)], dtype=np.float32)
    speeds = base._speed_array(frames)
    horizons = np.asarray(args.horizons, dtype=np.float32)
    label_cameras = list(args.label_cameras)

    frame_indices, camera_names, image_paths = _flatten_camera_paths(episode_dir, frames, label_cameras)
    if args.camera_teacher == "yolo":
        flat_labels = _yolo_multicam_labels(getattr(args, "_yolo_model", None), image_paths, camera_names, args)
    elif args.camera_teacher == "color":
        flat_labels = _color_multicam_labels(image_paths, camera_names, args)
    else:
        flat_labels = [_empty_camera_label(camera_name) for camera_name in camera_names]
    camera_labels_by_frame = _group_labels_by_frame(frame_indices, flat_labels, len(frames))

    stop_states = []
    front_obstacles = []
    for idx, frame in enumerate(frames):
        future_idx = base._future_indices(times, idx, horizons)
        stop_states.append(base._stop_state_label(float(speeds[idx]), speeds[future_idx], args))
        lidar_token = frame.get("lidar_bev_token")
        front_obstacles.append(base._bev_front_obstacle(episode_dir / lidar_token, args) if lidar_token else {"valid": False, "confidence": 0.0})

    output_path = episode_dir / args.output_name
    counts = {
        "frames": 0,
        "stop_sign": 0,
        "traffic_light": 0,
        "front_vehicle": 0,
        "reason_labeled": 0,
    }
    state_counts = {name: 0 for name in STOP_STATE_NAMES}
    reason_counts = {name: 0 for name in STOP_REASON_NAMES}
    with output_path.open("w", encoding="utf-8", buffering=1) as handle:
        for idx, frame in enumerate(frames):
            stop_state = stop_states[idx]
            lo = int(np.searchsorted(times, times[idx] - args.reason_pre_sec))
            hi = int(np.searchsorted(times, times[idx] + args.reason_post_sec, side="right"))
            context_labels = _flatten_labels(camera_labels_by_frame[lo:hi])
            context_obstacles = front_obstacles[lo:hi]

            stop_sign = base._best_detection(context_labels, "stop_sign")
            traffic_light = base._best_detection(context_labels, "traffic_light", valid_states=("red", "yellow"))
            if not traffic_light.get("valid"):
                traffic_light = base._best_detection(camera_labels_by_frame[idx], "traffic_light")
            front_obstacle = base._best_detection([{"front_vehicle": item} for item in context_obstacles], "front_vehicle")
            reason = base._choose_reason(frame, stop_state, traffic_light, stop_sign, front_obstacle, args)

            label = {
                "episode_token": frame.get("episode_token"),
                "frame_token": frame.get("frame_token"),
                "step": int(frame.get("step", idx)),
                "time": float(frame.get("time", 0.0)),
                "source": "teach2drive_pseudo_label_multicam_v1",
                "label_cameras": label_cameras,
                "stop_state": {
                    "name": stop_state,
                    "id": STOP_STATE_NAMES.index(stop_state),
                    "source": "odom_temporal_teacher",
                },
                "pseudo_stop_reason": reason,
                "traffic_light": traffic_light,
                "stop_sign": stop_sign,
                "front_vehicle": {
                    "valid": bool(front_obstacle.get("valid", False)),
                    "distance_m": front_obstacle.get("distance_m"),
                    "confidence": float(front_obstacle.get("confidence", 0.0)),
                    "source": front_obstacle.get("source", "lidar_bev_front_obstacle_teacher"),
                },
                "camera_detections": camera_labels_by_frame[idx],
                "lane": {"valid": False, "source": "not_estimated"},
            }
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")
            counts["frames"] += 1
            state_counts[stop_state] += 1
            reason_counts[reason["name"]] += 1
            counts["reason_labeled"] += int(float(reason.get("mask", 0.0)) > 0.0)
            counts["stop_sign"] += int(bool(stop_sign.get("valid", False)))
            counts["traffic_light"] += int(bool(traffic_light.get("valid", False)))
            counts["front_vehicle"] += int(bool(front_obstacle.get("valid", False)))

    return {
        "episode_dir": str(episode_dir),
        "output": str(output_path),
        "label_cameras": label_cameras,
        **counts,
        "stop_state_counts": state_counts,
        "stop_reason_counts": reason_counts,
    }


def label_dataset(args: argparse.Namespace) -> None:
    episode_dirs = base._discover_episode_dirs(args.input_root)
    if not episode_dirs:
        raise RuntimeError(f"No tokenized episodes found under: {args.input_root}")
    args._yolo_model = base._load_yolo_model(args)
    summaries = []
    for episode_idx, episode_dir in enumerate(episode_dirs):
        summary = _label_episode(episode_dir, args)
        summaries.append(summary)
        print(json.dumps({"episode": episode_idx + 1, "total": len(episode_dirs), **summary}, ensure_ascii=False), flush=True)
    dataset_summary = {
        "source": "teach2drive_pseudo_label_multicam_v1",
        "label_cameras": list(args.label_cameras),
        "episodes": summaries,
        "total_frames": int(sum(item["frames"] for item in summaries)),
        "total_reason_labeled": int(sum(item["reason_labeled"] for item in summaries)),
        "total_stop_sign": int(sum(item["stop_sign"] for item in summaries)),
        "total_traffic_light": int(sum(item["traffic_light"] for item in summaries)),
        "total_front_vehicle": int(sum(item["front_vehicle"] for item in summaries)),
    }
    out = Path(args.summary_output).expanduser() if args.summary_output else Path(args.input_root[0]).expanduser() / "pseudo_label_multicam_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = base.build_arg_parser()
    parser.description = "Generate multicamera post-hoc pseudo labels for tokenized Teach2Drive episodes."
    parser.set_defaults(output_name="pseudo_labels_multicam.jsonl")
    parser.add_argument("--label-cameras", type=_parse_csv, default=["front", "left", "right"])
    return parser


def main() -> None:
    label_dataset(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
