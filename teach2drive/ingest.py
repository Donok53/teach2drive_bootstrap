import argparse
import json
import math
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import quat_to_yaw


PREFERRED_ODOM_TOPICS = (
    "/odom",
    "/wheel/odometry",
    "/lio_localizer/odometry/optimization",
    "/odometry/filtered",
)
PREFERRED_IMU_TOPICS = ("/imu/data", "/imu", "/ouster/imu")
PREFERRED_CAMERA_TOPICS = (
    "/camera/image_raw",
    "/camera/color/image_raw",
    "/front_camera/image_raw",
    "/usb_cam/image_raw",
)
PREFERRED_LIDAR_TOPICS = (
    "/velodyne_points",
    "/ouster/points",
    "/points_raw",
    "/livox/lidar",
    "/scan",
)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required when decoding/writing camera frames. Install python3-opencv "
            "for system ROS Python or opencv-python-headless for a virtualenv."
        ) from exc
    return cv2


def _points_to_bev(points, grid_size, x_min, x_max, y_min, y_max, z_min, z_max):
    occ = np.zeros((grid_size, grid_size), dtype=np.float32)
    height = np.zeros((grid_size, grid_size), dtype=np.float32)
    intensity = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)
    if points.size == 0:
        return np.stack([occ, height, intensity], axis=0).astype(np.float16)

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    valid = (
        np.isfinite(xs)
        & np.isfinite(ys)
        & np.isfinite(zs)
        & (xs >= x_min)
        & (xs < x_max)
        & (ys >= y_min)
        & (ys < y_max)
        & (zs >= z_min)
        & (zs <= z_max)
    )
    if not np.any(valid):
        return np.stack([occ, height, intensity], axis=0).astype(np.float16)

    xs = xs[valid]
    ys = ys[valid]
    zs = zs[valid]
    inten = points[:, 3][valid] if points.shape[1] > 3 else np.ones_like(xs)
    ix = np.clip(((xs - x_min) * grid_size / max(x_max - x_min, 1e-6)).astype(np.int32), 0, grid_size - 1)
    iy = np.clip(((ys - y_min) * grid_size / max(y_max - y_min, 1e-6)).astype(np.int32), 0, grid_size - 1)
    rows = grid_size - 1 - ix
    cols = iy
    flat = rows * grid_size + cols

    np.add.at(occ.reshape(-1), flat, 1.0)
    np.add.at(count.reshape(-1), flat, 1.0)
    np.maximum.at(height.reshape(-1), flat, (zs - z_min) / max(z_max - z_min, 1e-6))
    np.add.at(intensity.reshape(-1), flat, inten)

    occ = np.clip(np.log1p(occ) / np.log(16.0), 0.0, 1.0)
    nonzero = count > 0
    if np.any(nonzero):
        intensity[nonzero] = intensity[nonzero] / count[nonzero]
        hi = np.percentile(intensity[nonzero], 95)
        if hi > 1e-6:
            intensity = np.clip(intensity / hi, 0.0, 1.0)
    return np.stack([occ, height, intensity], axis=0).astype(np.float16)


def _topic_table(bag) -> Dict[str, str]:
    info = bag.get_type_and_topic_info()
    return {topic: meta.msg_type for topic, meta in info.topics.items()}


def _choose_topic(
    topic_types: Dict[str, str],
    requested: str,
    msg_types: Sequence[str],
    preferred: Sequence[str],
    used: Optional[set] = None,
) -> Optional[str]:
    if requested == "none":
        return None
    if requested != "auto":
        return requested if requested in topic_types else None
    used = used or set()
    candidates = [topic for topic, typ in topic_types.items() if typ in msg_types and topic not in used]
    for topic in preferred:
        if topic in candidates:
            return topic
    return sorted(candidates)[0] if candidates else None


def _parse_camera_topics(text: str) -> Dict[str, str]:
    mapping = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError("Camera topics must look like front:auto,left:/topic,right:none")
        name, topic = item.split(":", 1)
        name = name.strip()
        topic = topic.strip()
        if not name:
            raise argparse.ArgumentTypeError("Camera name cannot be empty.")
        mapping[name] = topic or "none"
    if not mapping:
        raise argparse.ArgumentTypeError("At least one camera mapping is required.")
    return mapping


def _stamp_sec(stamp) -> float:
    return float(stamp.to_sec())


def _decode_compressed_image(msg, image_size: Tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode CompressedImage.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != tuple(image_size):
        rgb = cv2.resize(rgb, tuple(image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


def _decode_image_msg(msg, image_size: Tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    height, width = int(msg.height), int(msg.width)
    enc = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in {"rgb8", "bgr8"}:
        img = raw.reshape(height, msg.step)[:, : width * 3].reshape(height, width, 3)
        if enc == "bgr8":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif enc in {"rgba8", "bgra8"}:
        img = raw.reshape(height, msg.step)[:, : width * 4].reshape(height, width, 4)
        code = cv2.COLOR_RGBA2RGB if enc == "rgba8" else cv2.COLOR_BGRA2RGB
        img = cv2.cvtColor(img, code)
    elif enc in {"mono8", "8uc1"}:
        gray = raw.reshape(height, msg.step)[:, :width]
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")
    if (width, height) != tuple(image_size):
        img = cv2.resize(img, tuple(image_size), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def _decode_camera(msg, msg_type: str, image_size: Tuple[int, int]) -> np.ndarray:
    if msg_type == "sensor_msgs/CompressedImage":
        return _decode_compressed_image(msg, image_size)
    return _decode_image_msg(msg, image_size)


def _cloud_to_bev(
    msg,
    grid_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    from sensor_msgs.msg import PointField

    dtype_map = {
        PointField.INT8: np.int8,
        PointField.UINT8: np.uint8,
        PointField.INT16: np.int16,
        PointField.UINT16: np.uint16,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }
    names = []
    formats = []
    offsets = []
    for field in msg.fields:
        if field.name in {"x", "y", "z", "intensity"}:
            names.append(field.name)
            formats.append(dtype_map[field.datatype])
            offsets.append(field.offset)
    if not {"x", "y", "z"}.issubset(set(names)):
        return np.zeros((3, grid_size, grid_size), dtype=np.float16)
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    raw = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    xs = raw["x"].astype(np.float32, copy=False)
    ys = raw["y"].astype(np.float32, copy=False)
    zs = raw["z"].astype(np.float32, copy=False)
    if "intensity" in raw.dtype.names:
        intensity = raw["intensity"].astype(np.float32, copy=False)
    else:
        intensity = np.ones_like(xs, dtype=np.float32)
    points = np.stack([xs, ys, zs, intensity], axis=1)
    return _points_to_bev(points, grid_size, x_min, x_max, y_min, y_max, z_min, z_max)


def _scan_to_bev(msg, args) -> np.ndarray:
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    angles = float(msg.angle_min) + np.arange(len(ranges), dtype=np.float32) * float(msg.angle_increment)
    valid = np.isfinite(ranges) & (ranges >= float(msg.range_min)) & (ranges <= float(msg.range_max))
    if not np.any(valid):
        return np.zeros((3, args.bev_size, args.bev_size), dtype=np.float16)
    rs = ranges[valid]
    ang = angles[valid]
    points = np.stack([rs * np.cos(ang), rs * np.sin(ang), np.zeros_like(rs), np.ones_like(rs)], axis=1)
    return _points_to_bev(points, args.bev_size, args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)


def _decode_lidar(msg, msg_type: str, args) -> np.ndarray:
    if msg_type == "sensor_msgs/LaserScan":
        return _scan_to_bev(msg, args)
    return _cloud_to_bev(msg, args.bev_size, args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.searchsorted(source_times, target_times)
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(source_times[prev] - target_times) < np.abs(source_times[idx] - target_times)
    nearest = np.where(choose_prev, prev, idx)
    gap = np.abs(source_times[nearest] - target_times)
    return nearest.astype(np.int64), gap.astype(np.float32)


def _assign_nearest(target_times: np.ndarray, t: float, max_gap: float) -> Tuple[Optional[int], float]:
    idx = int(np.searchsorted(target_times, t))
    candidates = []
    if 0 <= idx < len(target_times):
        candidates.append(idx)
    if 0 <= idx - 1 < len(target_times):
        candidates.append(idx - 1)
    if not candidates:
        return None, float("inf")
    best = min(candidates, key=lambda item: abs(float(target_times[item]) - t))
    gap = abs(float(target_times[best]) - t)
    if gap > max_gap:
        return None, gap
    return best, gap


def _derive_motion_if_needed(times: np.ndarray, x: np.ndarray, y: np.ndarray, yaw: np.ndarray, v: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(times) < 2:
        return v, w
    dt = np.gradient(times)
    dt[dt < 1e-3] = 1e-3
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    signed_speed = (dx * heading[:, 0] + dy * heading[:, 1]) / dt
    yaw_unwrapped = np.unwrap(yaw)
    yaw_rate = np.gradient(yaw_unwrapped) / dt
    if np.nanmax(np.abs(v)) < 1e-4 and np.nanmax(np.abs(signed_speed)) > 1e-4:
        v = signed_speed.astype(np.float32)
    if np.nanmax(np.abs(w)) < 1e-4 and np.nanmax(np.abs(yaw_rate)) > 1e-4:
        w = yaw_rate.astype(np.float32)
    return v.astype(np.float32), w.astype(np.float32)


def _derive_phases(v: np.ndarray, args) -> List[str]:
    moving = np.abs(v) > args.phase_move_speed_mps
    if not np.any(moving):
        return ["drive"] * len(v)
    first_move = int(np.argmax(moving))
    last_move = len(moving) - 1 - int(np.argmax(moving[::-1]))
    phases = []
    for idx in range(len(v)):
        if args.derive_phases and idx < first_move:
            phases.append("stopped_start")
        elif args.derive_phases and idx > last_move:
            phases.append("stopped_end")
        else:
            phases.append("drive")
    return phases


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _prepare_output(root: Path, cameras: Sequence[str], overwrite: bool, episode_index: int) -> Path:
    episode_dir = root / f"episode_{episode_index:06d}"
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"Output episode already exists: {episode_dir}. Use --overwrite to replace it.")
    if episode_dir.exists() and overwrite:
        shutil.rmtree(episode_dir)
    (episode_dir / "lidar_bev").mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        (episode_dir / "camera" / camera).mkdir(parents=True, exist_ok=True)
    return episode_dir


def _discover_ros1_bags(source: Path) -> List[Path]:
    if source.is_file():
        if source.suffix != ".bag":
            raise RuntimeError(f"Expected a .bag file, got: {source}")
        return [source]
    if source.is_dir():
        bags = sorted(source.rglob("*.bag"))
        if bags:
            return bags
    raise RuntimeError(f"No ROS1 bag files found in {source}")


def _inspect_ros1_bag(args) -> None:
    import rosbag

    bags = _discover_ros1_bags(Path(args.input).expanduser())
    inspect_bag = bags[0]
    with rosbag.Bag(str(inspect_bag)) as bag:
        topic_types = _topic_table(bag)
    print(json.dumps({
        "input": str(Path(args.input).expanduser()),
        "bag_count": len(bags),
        "inspected_bag": str(inspect_bag),
        "topics": [{"name": topic, "type": typ} for topic, typ in sorted(topic_types.items())],
        "recommended": {
            "odom_topic": _choose_topic(topic_types, "auto", ("nav_msgs/Odometry",), PREFERRED_ODOM_TOPICS),
            "imu_topic": _choose_topic(topic_types, "auto", ("sensor_msgs/Imu",), PREFERRED_IMU_TOPICS),
            "front_camera": _choose_topic(topic_types, "auto", ("sensor_msgs/Image", "sensor_msgs/CompressedImage"), PREFERRED_CAMERA_TOPICS),
            "lidar_topic": _choose_topic(topic_types, "auto", ("sensor_msgs/PointCloud2", "sensor_msgs/LaserScan"), PREFERRED_LIDAR_TOPICS),
        },
    }, ensure_ascii=False, indent=2))


def _ingest_single_ros1_bag(args, bag_path: Path, output_root: Path, episode_index: int) -> Tuple[Dict, Dict]:
    import rosbag

    camera_requests = args.camera_topics
    cameras = list(camera_requests)
    cv2 = _cv2() if cameras else None
    episode_dir = _prepare_output(output_root, cameras, args.overwrite, episode_index)

    with rosbag.Bag(str(bag_path)) as bag:
        topic_types = _topic_table(bag)
        used_camera_topics = set()
        camera_topics = {}
        for name, requested in camera_requests.items():
            topic = _choose_topic(
                topic_types,
                requested,
                ("sensor_msgs/Image", "sensor_msgs/CompressedImage"),
                PREFERRED_CAMERA_TOPICS,
                used_camera_topics,
            )
            camera_topics[name] = topic
            if topic is not None:
                used_camera_topics.add(topic)
        odom_topic = _choose_topic(topic_types, args.odom_topic, ("nav_msgs/Odometry",), PREFERRED_ODOM_TOPICS)
        imu_topic = _choose_topic(topic_types, args.imu_topic, ("sensor_msgs/Imu",), PREFERRED_IMU_TOPICS)
        lidar_topic = _choose_topic(topic_types, args.lidar_topic, ("sensor_msgs/PointCloud2", "sensor_msgs/LaserScan"), PREFERRED_LIDAR_TOPICS)
        if odom_topic is None:
            raise RuntimeError("No odometry topic found. Pass --odom-topic explicitly.")

        odom_rows = []
        for _topic, msg, stamp in bag.read_messages(topics=[odom_topic]):
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            tw = msg.twist.twist
            odom_rows.append((_stamp_sec(stamp), float(p.x), float(p.y), quat_to_yaw(q.x, q.y, q.z, q.w), float(tw.linear.x), float(tw.angular.z)))

        if len(odom_rows) < 2:
            raise RuntimeError(f"Odometry topic {odom_topic} produced too few messages: {len(odom_rows)}")

        odom = np.asarray(odom_rows, dtype=np.float64)
        odom = odom[np.argsort(odom[:, 0])]
        start_time = float(odom[0, 0]) + max(0.0, float(args.start_offset_sec))
        end_time = float(odom[-1, 0])
        if args.max_duration_sec is not None:
            end_time = min(end_time, start_time + max(0.0, float(args.max_duration_sec)))
        if end_time <= start_time:
            raise RuntimeError("Selected bag time window is empty. Check --start-offset-sec and --max-duration-sec.")
        target_abs = np.arange(start_time, end_time + 1e-6, 1.0 / args.hz, dtype=np.float64)
        nearest_odom, odom_gap = _nearest_indices(odom[:, 0], target_abs)
        odom_valid = odom_gap <= args.max_odom_gap
        valid_indices = np.nonzero(odom_valid)[0]
        if len(valid_indices) < 2:
            raise RuntimeError("Too few odom-aligned frames. Increase --max-odom-gap or check --hz.")

        x = odom[nearest_odom, 1].astype(np.float32)
        y = odom[nearest_odom, 2].astype(np.float32)
        yaw = odom[nearest_odom, 3].astype(np.float32)
        v = odom[nearest_odom, 4].astype(np.float32)
        w = odom[nearest_odom, 5].astype(np.float32)
        v, w = _derive_motion_if_needed(target_abs - start_time, x, y, yaw, v, w)
        phases = _derive_phases(v, args)

        camera_best_gap = {name: np.full(len(target_abs), np.inf, dtype=np.float32) for name in cameras}
        camera_valid = {name: np.zeros(len(target_abs), dtype=bool) for name in cameras}
        lidar_best_gap = np.full(len(target_abs), np.inf, dtype=np.float32)
        lidar_valid = np.zeros(len(target_abs), dtype=bool)
        imu_best_gap = np.full(len(target_abs), np.inf, dtype=np.float32)
        imu_valid = np.zeros(len(target_abs), dtype=bool)
        imu_values = np.zeros((len(target_abs), 6), dtype=np.float32)

        read_topics = [topic for topic in [imu_topic, lidar_topic, *camera_topics.values()] if topic is not None]
        read_topics = sorted(set(read_topics))
        topic_to_camera = {topic: name for name, topic in camera_topics.items() if topic is not None}

        for topic, msg, stamp in bag.read_messages(topics=read_topics):
            t = _stamp_sec(stamp)
            if topic == imu_topic:
                idx, gap = _assign_nearest(target_abs, t, args.max_imu_gap)
                if idx is not None and gap < imu_best_gap[idx]:
                    la = msg.linear_acceleration
                    av = msg.angular_velocity
                    imu_values[idx] = np.asarray([la.x, la.y, la.z, av.x, av.y, av.z], dtype=np.float32)
                    imu_valid[idx] = True
                    imu_best_gap[idx] = gap
            elif topic == lidar_topic:
                idx, gap = _assign_nearest(target_abs, t, args.max_lidar_gap)
                if idx is not None and gap < lidar_best_gap[idx]:
                    lidar_bev = _decode_lidar(msg, topic_types[topic], args)
                    np.save(episode_dir / "lidar_bev" / f"{idx:06d}.npy", lidar_bev.astype(np.float16))
                    lidar_valid[idx] = True
                    lidar_best_gap[idx] = gap
            elif topic in topic_to_camera:
                idx, gap = _assign_nearest(target_abs, t, args.max_image_gap)
                camera_name = topic_to_camera[topic]
                if idx is not None and gap < camera_best_gap[camera_name][idx]:
                    rgb = _decode_camera(msg, topic_types[topic], tuple(args.image_size))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    path = episode_dir / "camera" / camera_name / f"{idx:06d}.jpg"
                    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                    if not ok:
                        raise RuntimeError(f"Failed to write image: {path}")
                    camera_valid[camera_name][idx] = True
                    camera_best_gap[camera_name][idx] = gap

    zero_lidar = np.zeros((3, args.bev_size, args.bev_size), dtype=np.float16)
    black_image = np.zeros((args.image_size[1], args.image_size[0], 3), dtype=np.uint8)
    frames_path = episode_dir / "frames.jsonl"
    episode_token = uuid.uuid4().hex
    written = 0
    with frames_path.open("w", encoding="utf-8", buffering=1) as handle:
        for step, idx in enumerate(valid_indices.tolist()):
            frame_token = uuid.uuid4().hex
            camera_tokens = {}
            camera_valid_record = {}
            for camera in cameras:
                rel = Path("camera") / camera / f"{idx:06d}.jpg"
                path = episode_dir / rel
                if not path.exists():
                    cv2.imwrite(str(path), black_image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
                camera_tokens[camera] = str(rel)
                camera_valid_record[camera] = bool(camera_valid[camera][idx])
            lidar_rel = Path("lidar_bev") / f"{idx:06d}.npy"
            lidar_path = episode_dir / lidar_rel
            if not lidar_path.exists():
                np.save(lidar_path, zero_lidar)
            record = {
                "episode_token": episode_token,
                "frame_token": frame_token,
                "step": int(step),
                "phase": phases[idx],
                "time": float(target_abs[idx] - start_time),
                "camera_tokens": camera_tokens,
                "lidar_bev_token": str(lidar_rel),
                "sensor_valid": {
                    "camera": camera_valid_record,
                    "image": bool(any(camera_valid_record.values())),
                    "lidar": bool(lidar_valid[idx]),
                    "imu": bool(imu_valid[idx]),
                },
                "imu": {
                    "accelerometer": [float(vv) for vv in imu_values[idx, :3]],
                    "gyroscope": [float(vv) for vv in imu_values[idx, 3:]],
                },
                "odom": {
                    "x": float(x[idx]),
                    "y": float(y[idx]),
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": float(yaw[idx]),
                    "v_forward": float(v[idx]),
                    "velocity": [float(v[idx] * math.cos(yaw[idx])), float(v[idx] * math.sin(yaw[idx])), 0.0],
                    "yaw_rate": float(w[idx]),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    dataset_meta = {
        "dataset": "teach2drive_tokenized_ingest",
        "source_type": "ros1_bag",
        "source": str(bag_path),
        "hz": float(args.hz),
        "cameras": cameras,
        "image_size_wh": args.image_size,
        "jpeg_quality": args.jpeg_quality,
        "lidar_bev_size": args.bev_size,
        "topics": {
            "odom": odom_topic,
            "imu": imu_topic,
            "lidar": lidar_topic,
            "cameras": camera_topics,
        },
        "notes": "cmd_vel/action topics are intentionally not used as policy targets.",
    }
    summary = {
        "episode_index": episode_index,
        "episode_token": episode_token,
        "source": str(bag_path),
        "frames": int(written),
        "duration_sec": float(target_abs[valid_indices[-1]] - target_abs[valid_indices[0]]) if len(valid_indices) else 0.0,
        "odom_topic": odom_topic,
        "imu_topic": imu_topic,
        "lidar_topic": lidar_topic,
        "camera_topics": camera_topics,
        "camera_coverage": {
            name: float(np.mean(camera_valid[name][valid_indices])) if len(valid_indices) else 0.0
            for name in cameras
        },
        "lidar_coverage": float(np.mean(lidar_valid[valid_indices])) if len(valid_indices) else 0.0,
        "imu_coverage": float(np.mean(imu_valid[valid_indices])) if len(valid_indices) else 0.0,
    }
    _write_json(episode_dir / "episode_meta.json", {**dataset_meta, "episode_token": episode_token, "episode_index": episode_index})
    _write_json(episode_dir / "episode_summary.json", summary)
    return dataset_meta, summary


def ingest_ros1_bag(args) -> None:
    source = Path(args.input).expanduser()
    output_root = Path(args.output).expanduser()
    bag_paths = _discover_ros1_bags(source)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    summaries = []
    root_meta = None
    for episode_index, bag_path in enumerate(bag_paths):
        meta, summary = _ingest_single_ros1_bag(args, bag_path, output_root, episode_index)
        summaries.append(summary)
        if root_meta is None:
            root_meta = dict(meta)

    if root_meta is None:
        raise RuntimeError("No episodes were written.")
    root_meta.update({
        "source": str(source),
        "sources": [str(path) for path in bag_paths],
        "source_type": "ros1_bag_dir" if source.is_dir() else "ros1_bag",
        "episode_count": len(summaries),
    })
    _write_json(output_root / "dataset_meta.json", root_meta)
    _write_json(output_root / "dataset_summary.json", {"episodes": summaries})
    print(json.dumps({
        "output_root": str(output_root),
        "episode_count": len(summaries),
        "total_frames": int(sum(item["frames"] for item in summaries)),
        "episodes": summaries,
    }, ensure_ascii=False, indent=2))


def ingest_existing_token_dataset(args) -> None:
    src = Path(args.input).expanduser()
    if not (src / "dataset_meta.json").exists() and not any(src.glob("episode_*/frames.jsonl")):
        raise RuntimeError(f"Input folder does not look like a Teach2Drive token dataset: {src}")
    out = Path(args.output).expanduser()
    if out.resolve() == src.resolve():
        print(json.dumps({"status": "already_tokenized", "output_root": str(src)}, indent=2))
        return
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out}. Use --overwrite to replace it.")
    if args.link_existing:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.symlink_to(src, target_is_directory=True)
    else:
        shutil.copytree(src, out)
    print(json.dumps({"status": "copied_token_dataset", "input": str(src), "output_root": str(out)}, indent=2))


def ingest(args) -> None:
    source = Path(args.input).expanduser()
    input_type = args.input_type
    if input_type == "auto":
        if source.suffix == ".bag":
            input_type = "ros1_bag"
        elif source.is_dir():
            if (source / "dataset_meta.json").exists() or any(source.glob("episode_*/frames.jsonl")):
                input_type = "token_dataset"
            elif any(source.rglob("*.bag")):
                input_type = "ros1_bag"
            else:
                input_type = "token_dataset"
        else:
            raise RuntimeError(f"Could not infer input type for {source}. Pass --input-type explicitly.")
    if args.inspect:
        if input_type != "ros1_bag":
            raise RuntimeError("--inspect is currently supported for ROS1 bags only.")
        _inspect_ros1_bag(args)
        return
    if input_type == "ros1_bag":
        ingest_ros1_bag(args)
    elif input_type == "token_dataset":
        ingest_existing_token_dataset(args)
    else:
        raise RuntimeError(f"Unsupported input type: {input_type}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize heterogeneous logs into Teach2Drive token episode format.")
    parser.add_argument("--input", required=True, help="Input ROS1 bag or existing token dataset folder.")
    parser.add_argument("--output", required=True, help="Output token dataset root.")
    parser.add_argument("--input-type", choices=["auto", "ros1_bag", "token_dataset"], default="auto")
    parser.add_argument("--inspect", action="store_true", help="Print detected topics and exit.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-existing", action="store_true", help="For token_dataset inputs, symlink instead of copying.")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--start-offset-sec", type=float, default=0.0, help="Skip this many seconds from the first odom stamp.")
    parser.add_argument("--max-duration-sec", type=float, default=None, help="Optional short conversion window for smoke tests.")
    parser.add_argument("--odom-topic", default="auto")
    parser.add_argument("--imu-topic", default="auto")
    parser.add_argument("--lidar-topic", default="auto")
    parser.add_argument("--camera-topics", type=_parse_camera_topics, default=_parse_camera_topics("front:auto"))
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-min", type=float, default=-14.0)
    parser.add_argument("--y-max", type=float, default=14.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--max-odom-gap", type=float, default=0.15)
    parser.add_argument("--max-image-gap", type=float, default=0.15)
    parser.add_argument("--max-lidar-gap", type=float, default=0.20)
    parser.add_argument("--max-imu-gap", type=float, default=0.08)
    parser.add_argument("--no-derive-phases", dest="derive_phases", action="store_false")
    parser.set_defaults(derive_phases=True)
    parser.add_argument("--phase-move-speed-mps", type=float, default=0.25)
    return parser


def main() -> None:
    ingest(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
