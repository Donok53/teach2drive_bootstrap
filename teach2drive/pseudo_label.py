import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .token_dataset import STOP_REASON_NAMES, STOP_STATE_NAMES


def _read_frames(path: Path) -> List[Dict]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    frames.sort(key=lambda item: (int(item.get("step", 0)), float(item.get("time", 0.0))))
    return frames


def _discover_episode_dirs(input_roots: Sequence[str]) -> List[Path]:
    episode_dirs = []
    for root_text in input_roots:
        root = Path(root_text).expanduser()
        if (root / "frames.jsonl").exists():
            episode_dirs.append(root.resolve())
            continue
        episode_dirs.extend(sorted(path.resolve() for path in root.glob("episode_*") if (path / "frames.jsonl").exists()))
    return sorted(dict.fromkeys(episode_dirs))


def _speed_array(frames: Sequence[Dict]) -> np.ndarray:
    return np.asarray([float(frame.get("odom", {}).get("v_forward", 0.0)) for frame in frames], dtype=np.float32)


def _future_indices(times: np.ndarray, idx: int, horizons: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, times[idx] + horizons)
    return np.clip(indices, 0, len(times) - 1)


def _stop_state_label(current_speed: float, future_speeds: np.ndarray, args: argparse.Namespace) -> str:
    current_speed = abs(float(current_speed))
    future_abs = np.abs(future_speeds.astype(np.float32))
    future_min = float(np.min(future_abs))
    future_max = float(np.max(future_abs))
    if current_speed <= args.stop_speed_mps:
        if future_max >= args.move_speed_mps:
            return "release_go"
        return "stopped_waiting"
    if future_min <= args.stop_speed_mps:
        return "approach_stop"
    return "drive"


def _hsv_red_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.asarray([0, 65, 55], dtype=np.uint8)
    upper1 = np.asarray([12, 255, 255], dtype=np.uint8)
    lower2 = np.asarray([168, 65, 55], dtype=np.uint8)
    upper2 = np.asarray([180, 255, 255], dtype=np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))


def _detect_stop_sign(image_path: Path, args: argparse.Namespace) -> Dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"valid": False, "confidence": 0.0, "source": "camera_red_shape_teacher"}
    height, width = image.shape[:2]
    mask = _hsv_red_mask(image)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < args.stop_sign_min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if y > height * args.stop_sign_max_y_ratio:
            continue
        aspect = w / max(h, 1)
        if not (0.55 <= aspect <= 1.65):
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        fill_ratio = area / max(float(w * h), 1.0)
        shape_score = 1.0 - min(abs(len(approx) - 8), 6) / 6.0
        area_score = min(area / max(args.stop_sign_ref_area_px, 1.0), 1.0)
        confidence = float(np.clip(0.45 * fill_ratio + 0.35 * shape_score + 0.20 * area_score, 0.0, 1.0))
        candidate = {
            "valid": confidence >= args.stop_sign_confidence,
            "confidence": confidence,
            "bbox_xywh": [int(x), int(y), int(w), int(h)],
            "image": str(image_path.name),
            "source": "camera_red_shape_teacher",
            "distance_m": None,
        }
        if best is None or candidate["confidence"] > best["confidence"]:
            best = candidate
    return best or {"valid": False, "confidence": 0.0, "source": "camera_red_shape_teacher", "distance_m": None}


def _detect_traffic_light(image_path: Path, args: argparse.Namespace) -> Dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {"valid": False, "state": "Unknown", "confidence": 0.0, "source": "camera_color_blob_teacher"}
    height, width = image.shape[:2]
    roi = image[: int(height * args.traffic_light_max_y_ratio), :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ranges = {
        "Red": [
            (np.asarray([0, 100, 120], dtype=np.uint8), np.asarray([12, 255, 255], dtype=np.uint8)),
            (np.asarray([168, 100, 120], dtype=np.uint8), np.asarray([180, 255, 255], dtype=np.uint8)),
        ],
        "Yellow": [(np.asarray([18, 80, 120], dtype=np.uint8), np.asarray([38, 255, 255], dtype=np.uint8))],
        "Green": [(np.asarray([42, 60, 100], dtype=np.uint8), np.asarray([92, 255, 255], dtype=np.uint8))],
    }
    best = {"state": "Unknown", "confidence": 0.0, "valid": False, "source": "camera_color_blob_teacher"}
    for state, state_ranges in ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in state_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < args.traffic_light_min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if not (0.35 <= aspect <= 2.5):
                continue
            confidence = float(np.clip(area / max(args.traffic_light_ref_area_px, 1.0), 0.0, 1.0))
            if confidence > best["confidence"]:
                best = {
                    "valid": confidence >= args.traffic_light_confidence,
                    "state": state,
                    "confidence": confidence,
                    "bbox_xywh": [int(x), int(y), int(w), int(h)],
                    "image": str(image_path.name),
                    "source": "camera_color_blob_teacher",
                }
    return best


def _traffic_light_state_from_crop(image_bgr: np.ndarray) -> Dict:
    if image_bgr.size == 0:
        return {"state": "Unknown", "confidence": 0.0}
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    masks = {
        "Red": cv2.bitwise_or(
            cv2.inRange(hsv, np.asarray([0, 90, 100], dtype=np.uint8), np.asarray([12, 255, 255], dtype=np.uint8)),
            cv2.inRange(hsv, np.asarray([168, 90, 100], dtype=np.uint8), np.asarray([180, 255, 255], dtype=np.uint8)),
        ),
        "Yellow": cv2.inRange(hsv, np.asarray([18, 70, 100], dtype=np.uint8), np.asarray([38, 255, 255], dtype=np.uint8)),
        "Green": cv2.inRange(hsv, np.asarray([42, 55, 90], dtype=np.uint8), np.asarray([92, 255, 255], dtype=np.uint8)),
    }
    scores = {name: float(np.count_nonzero(mask)) for name, mask in masks.items()}
    state, score = max(scores.items(), key=lambda item: item[1])
    denom = max(float(image_bgr.shape[0] * image_bgr.shape[1]), 1.0)
    confidence = float(np.clip(score / denom * 8.0, 0.0, 1.0))
    if confidence <= 0.02:
        return {"state": "Unknown", "confidence": 0.0}
    return {"state": state, "confidence": confidence}


def _empty_camera_labels() -> Dict:
    return {
        "stop_sign": {"valid": False, "confidence": 0.0, "source": "no_camera_teacher", "distance_m": None},
        "traffic_light": {"valid": False, "state": "Unknown", "confidence": 0.0, "source": "no_camera_teacher"},
        "front_camera_vehicle": {"valid": False, "confidence": 0.0, "source": "no_camera_teacher"},
    }


def _load_yolo_model(args: argparse.Namespace):
    if args.camera_teacher != "yolo":
        return None
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("camera-teacher=yolo requires `pip install ultralytics`.") from exc
    return YOLO(args.yolo_model)


def _yolo_episode_labels(model, image_paths: Sequence[Path], args: argparse.Namespace) -> List[Dict]:
    if model is None:
        return [_empty_camera_labels() for _ in image_paths]
    labels = [_empty_camera_labels() for _ in image_paths]
    names = getattr(model, "names", {})
    for chunk_start in range(0, len(image_paths), args.yolo_chunk):
        chunk_paths = image_paths[chunk_start : chunk_start + args.yolo_chunk]
        valid_pairs = [(offset, path) for offset, path in enumerate(chunk_paths) if str(path) and path.exists()]
        if not valid_pairs:
            continue
        sources = [str(path) for _offset, path in valid_pairs]
        results = model.predict(
            source=sources,
            stream=False,
            imgsz=args.yolo_imgsz,
            conf=args.yolo_confidence,
            iou=args.yolo_iou,
            device=args.yolo_device or None,
            batch=args.yolo_batch,
            verbose=False,
        )
        for (offset, image_path), result in zip(valid_pairs, results):
            labels[chunk_start + offset] = _yolo_result_to_label(result, image_path, names, args)
    return labels


def _yolo_result_to_label(result, image_path: Path, names: Dict, args: argparse.Namespace) -> Dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    height, width = image.shape[:2] if image is not None else (1, 1)
    label = _empty_camera_labels()
    label["stop_sign"]["source"] = "yolo_teacher"
    label["traffic_light"]["source"] = "yolo_teacher"
    label["front_camera_vehicle"]["source"] = "yolo_teacher"
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return label
    for box in boxes:
        cls_id = int(box.cls.item())
        cls_name = str(names.get(cls_id, cls_id)).lower().replace("_", " ")
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy.cpu().numpy().reshape(-1)[:4]]
        bbox = [int(max(x1, 0)), int(max(y1, 0)), int(min(x2 - x1, width)), int(min(y2 - y1, height))]
        if cls_name == "stop sign" and confidence > label["stop_sign"]["confidence"]:
            label["stop_sign"] = {
                "valid": confidence >= args.yolo_stop_sign_confidence,
                "confidence": confidence,
                "bbox_xywh": bbox,
                "image": image_path.name,
                "source": "yolo_teacher",
                "distance_m": None,
            }
        elif cls_name == "traffic light" and confidence > label["traffic_light"]["confidence"]:
            state_info = {"state": "Unknown", "confidence": 0.0}
            if image is not None:
                xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
                xi2, yi2 = min(int(x2), width), min(int(y2), height)
                state_info = _traffic_light_state_from_crop(image[yi1:yi2, xi1:xi2])
            label["traffic_light"] = {
                "valid": confidence >= args.yolo_traffic_light_confidence,
                "state": state_info["state"],
                "confidence": confidence,
                "state_confidence": state_info["confidence"],
                "bbox_xywh": bbox,
                "image": image_path.name,
                "source": "yolo_teacher",
            }
        elif cls_name in {"car", "truck", "bus", "motorcycle"} and confidence > label["front_camera_vehicle"]["confidence"]:
            center_x = (x1 + x2) * 0.5 / max(width, 1)
            center_y = (y1 + y2) * 0.5 / max(height, 1)
            label["front_camera_vehicle"] = {
                "valid": confidence >= args.yolo_vehicle_confidence and args.vehicle_center_min_x <= center_x <= args.vehicle_center_max_x,
                "confidence": confidence,
                "bbox_xywh": bbox,
                "image": image_path.name,
                "center_xy": [float(center_x), float(center_y)],
                "class_name": cls_name,
                "source": "yolo_teacher",
            }
    return label


def _bev_front_obstacle(lidar_path: Path, args: argparse.Namespace) -> Dict:
    try:
        bev = np.load(lidar_path).astype(np.float32)
    except (FileNotFoundError, ValueError):
        return {"valid": False, "confidence": 0.0, "source": "lidar_bev_front_obstacle_teacher"}
    if bev.ndim != 3:
        return {"valid": False, "confidence": 0.0, "source": "lidar_bev_front_obstacle_teacher"}
    if bev.shape[0] >= 3:
        occ = bev[0]
        height = bev[1]
    else:
        occ = bev[..., 0]
        height = bev[..., 1] if bev.shape[-1] > 1 else np.zeros_like(occ)
    grid = occ.shape[0]
    xs = np.linspace(args.lidar_x_max, args.lidar_x_min, grid, endpoint=False) + (args.lidar_x_max - args.lidar_x_min) / max(grid * 2, 1)
    ys = np.linspace(args.lidar_y_min, args.lidar_y_max, grid, endpoint=False) + (args.lidar_y_max - args.lidar_y_min) / max(grid * 2, 1)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="ij")
    mask = (
        (occ >= args.front_obstacle_occ_threshold)
        & (height >= args.front_obstacle_height_threshold)
        & (x_grid >= args.front_obstacle_min_x_m)
        & (x_grid <= args.front_obstacle_max_x_m)
        & (np.abs(y_grid) <= args.front_obstacle_lateral_m)
    )
    count = int(np.count_nonzero(mask))
    if count <= 0:
        return {
            "valid": False,
            "confidence": 0.0,
            "source": "lidar_bev_front_obstacle_teacher",
            "distance_m": None,
        }
    distance = float(np.min(x_grid[mask]))
    confidence = float(np.clip(count / max(args.front_obstacle_ref_cells, 1), 0.0, 1.0))
    return {
        "valid": confidence >= args.front_obstacle_confidence,
        "confidence": confidence,
        "distance_m": distance,
        "occupied_cells": count,
        "source": "lidar_bev_front_obstacle_teacher",
    }


def _choose_reason(frame: Dict, stop_state: str, traffic_light: Dict, stop_sign: Dict, front_obstacle: Dict, args: argparse.Namespace) -> Dict:
    phase = str(frame.get("phase", "drive"))
    if stop_state == "drive":
        return {"name": "none", "id": STOP_REASON_NAMES.index("none"), "mask": 1.0, "confidence": 1.0, "source": "odom"}
    if phase == "stopped_start":
        return {"name": "startup", "id": STOP_REASON_NAMES.index("startup"), "mask": 1.0, "confidence": 1.0, "source": "phase"}
    if phase == "stopped_end":
        return {"name": "route_end", "id": STOP_REASON_NAMES.index("route_end"), "mask": 1.0, "confidence": 1.0, "source": "phase"}
    light_state = str(traffic_light.get("state", "")).lower()
    if traffic_light.get("valid") and light_state in {"red", "yellow"}:
        return {
            "name": "traffic_light",
            "id": STOP_REASON_NAMES.index("traffic_light"),
            "mask": 1.0,
            "confidence": float(traffic_light.get("confidence", 0.0)),
            "source": traffic_light.get("source", "pseudo"),
        }
    if stop_sign.get("valid"):
        return {
            "name": "stop_sign",
            "id": STOP_REASON_NAMES.index("stop_sign"),
            "mask": 1.0,
            "confidence": float(stop_sign.get("confidence", 0.0)),
            "source": stop_sign.get("source", "pseudo"),
        }
    if front_obstacle.get("valid") and front_obstacle.get("distance_m") is not None:
        if float(front_obstacle["distance_m"]) <= args.front_obstacle_reason_m:
            return {
                "name": "front_vehicle",
                "id": STOP_REASON_NAMES.index("front_vehicle"),
                "mask": 1.0,
                "confidence": float(front_obstacle.get("confidence", 0.0)),
                "source": front_obstacle.get("source", "pseudo"),
            }
    return {
        "name": "unknown_stop",
        "id": STOP_REASON_NAMES.index("unknown_stop"),
        "mask": float(args.train_unknown_reason),
        "confidence": 0.0,
        "source": "fallback",
    }


def _best_detection(items: Sequence[Dict], key: str, valid_states: Sequence[str] = ()) -> Dict:
    best = None
    for item in items:
        candidate = item.get(key, {})
        if not candidate.get("valid"):
            continue
        if valid_states and str(candidate.get("state", "")).lower() not in valid_states:
            continue
        if best is None or float(candidate.get("confidence", 0.0)) > float(best.get("confidence", 0.0)):
            best = candidate
    if best is not None:
        return best
    if key == "traffic_light":
        return {"valid": False, "state": "Unknown", "confidence": 0.0}
    return {"valid": False, "confidence": 0.0, "distance_m": None}


def _label_episode(episode_dir: Path, args: argparse.Namespace) -> Dict:
    frames = _read_frames(episode_dir / "frames.jsonl")
    times = np.asarray([float(frame.get("time", idx / max(args.hz, 1e-6))) for idx, frame in enumerate(frames)], dtype=np.float32)
    speeds = _speed_array(frames)
    horizons = np.asarray(args.horizons, dtype=np.float32)
    yolo_model = _load_yolo_model(args)
    front_image_paths = []
    for frame in frames:
        front_token = frame.get("camera_tokens", {}).get(args.front_camera)
        front_image_paths.append(episode_dir / front_token if front_token else Path(""))
    yolo_labels = _yolo_episode_labels(yolo_model, front_image_paths, args) if args.camera_teacher == "yolo" else None
    stop_states = []
    camera_labels = []
    front_obstacles = []
    for idx, frame in enumerate(frames):
        future_idx = _future_indices(times, idx, horizons)
        stop_states.append(_stop_state_label(float(speeds[idx]), speeds[future_idx], args))
        camera_tokens = frame.get("camera_tokens", {})
        front_token = camera_tokens.get(args.front_camera)
        front_image = episode_dir / front_token if front_token else None
        if yolo_labels is not None:
            camera_label = yolo_labels[idx]
            stop_sign = camera_label["stop_sign"]
            traffic_light = camera_label["traffic_light"]
        elif args.camera_teacher == "color" and front_image and front_image.exists():
            stop_sign = _detect_stop_sign(front_image, args)
            traffic_light = _detect_traffic_light(front_image, args)
        else:
            stop_sign = {"valid": False, "confidence": 0.0, "source": "camera_red_shape_teacher", "distance_m": None}
            traffic_light = {"valid": False, "state": "Unknown", "confidence": 0.0, "source": "camera_color_blob_teacher"}
        camera_labels.append({"stop_sign": stop_sign, "traffic_light": traffic_light})
        lidar_token = frame.get("lidar_bev_token")
        front_obstacles.append(_bev_front_obstacle(episode_dir / lidar_token, args) if lidar_token else {"valid": False, "confidence": 0.0})

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
            context_labels = camera_labels[lo:hi]
            context_obstacles = front_obstacles[lo:hi]
            stop_sign = _best_detection(context_labels, "stop_sign")
            traffic_light = _best_detection(context_labels, "traffic_light", valid_states=("red", "yellow"))
            if not traffic_light.get("valid"):
                traffic_light = camera_labels[idx]["traffic_light"]
            front_obstacle = _best_detection([{"front_vehicle": item} for item in context_obstacles], "front_vehicle")
            reason = _choose_reason(frame, stop_state, traffic_light, stop_sign, front_obstacle, args)
            label = {
                "episode_token": frame.get("episode_token"),
                "frame_token": frame.get("frame_token"),
                "step": int(frame.get("step", idx)),
                "time": float(frame.get("time", 0.0)),
                "source": "teach2drive_pseudo_label_v1",
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
        **counts,
        "stop_state_counts": state_counts,
        "stop_reason_counts": reason_counts,
    }


def label_dataset(args: argparse.Namespace) -> None:
    episode_dirs = _discover_episode_dirs(args.input_root)
    if not episode_dirs:
        raise RuntimeError(f"No tokenized episodes found under: {args.input_root}")
    summaries = []
    for episode_idx, episode_dir in enumerate(episode_dirs):
        summary = _label_episode(episode_dir, args)
        summaries.append(summary)
        print(json.dumps({"episode": episode_idx + 1, "total": len(episode_dirs), **summary}, ensure_ascii=False), flush=True)
    dataset_summary = {
        "source": "teach2drive_pseudo_label_v1",
        "episodes": summaries,
        "total_frames": int(sum(item["frames"] for item in summaries)),
        "total_reason_labeled": int(sum(item["reason_labeled"] for item in summaries)),
        "total_stop_sign": int(sum(item["stop_sign"] for item in summaries)),
        "total_traffic_light": int(sum(item["traffic_light"] for item in summaries)),
        "total_front_vehicle": int(sum(item["front_vehicle"] for item in summaries)),
    }
    out = Path(args.summary_output).expanduser() if args.summary_output else Path(args.input_root[0]).expanduser() / "pseudo_label_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))


def _parse_horizons(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate post-hoc pseudo labels for tokenized Teach2Drive episodes.")
    parser.add_argument("--input-root", nargs="+", required=True)
    parser.add_argument("--output-name", default="pseudo_labels.jsonl")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--front-camera", default="front")
    parser.add_argument("--camera-teacher", choices=["none", "color", "yolo"], default="none")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--yolo-device", default="")
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-batch", type=int, default=16)
    parser.add_argument("--yolo-chunk", type=int, default=128)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.7)
    parser.add_argument("--yolo-stop-sign-confidence", type=float, default=0.30)
    parser.add_argument("--yolo-traffic-light-confidence", type=float, default=0.30)
    parser.add_argument("--yolo-vehicle-confidence", type=float, default=0.30)
    parser.add_argument("--vehicle-center-min-x", type=float, default=0.25)
    parser.add_argument("--vehicle-center-max-x", type=float, default=0.75)
    parser.add_argument("--horizons", type=_parse_horizons, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--stop-speed-mps", type=float, default=0.35)
    parser.add_argument("--move-speed-mps", type=float, default=1.0)
    parser.add_argument("--reason-pre-sec", type=float, default=4.0)
    parser.add_argument("--reason-post-sec", type=float, default=1.0)
    parser.add_argument("--train-unknown-reason", action="store_true")
    parser.add_argument("--stop-sign-min-area-px", type=float, default=80.0)
    parser.add_argument("--stop-sign-ref-area-px", type=float, default=900.0)
    parser.add_argument("--stop-sign-confidence", type=float, default=0.45)
    parser.add_argument("--stop-sign-max-y-ratio", type=float, default=0.85)
    parser.add_argument("--traffic-light-min-area-px", type=float, default=8.0)
    parser.add_argument("--traffic-light-ref-area-px", type=float, default=60.0)
    parser.add_argument("--traffic-light-confidence", type=float, default=0.35)
    parser.add_argument("--traffic-light-max-y-ratio", type=float, default=0.55)
    parser.add_argument("--lidar-x-min", type=float, default=-8.0)
    parser.add_argument("--lidar-x-max", type=float, default=20.0)
    parser.add_argument("--lidar-y-min", type=float, default=-14.0)
    parser.add_argument("--lidar-y-max", type=float, default=14.0)
    parser.add_argument("--front-obstacle-min-x-m", type=float, default=2.0)
    parser.add_argument("--front-obstacle-max-x-m", type=float, default=18.0)
    parser.add_argument("--front-obstacle-lateral-m", type=float, default=2.2)
    parser.add_argument("--front-obstacle-height-threshold", type=float, default=0.18)
    parser.add_argument("--front-obstacle-occ-threshold", type=float, default=0.05)
    parser.add_argument("--front-obstacle-ref-cells", type=int, default=32)
    parser.add_argument("--front-obstacle-confidence", type=float, default=0.25)
    parser.add_argument("--front-obstacle-reason-m", type=float, default=12.0)
    return parser


def main() -> None:
    label_dataset(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
