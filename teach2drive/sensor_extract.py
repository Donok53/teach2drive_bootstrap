import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import PointField

from .geometry import quat_to_yaw


def _topic_table(bag) -> Dict[str, str]:
    info = bag.get_type_and_topic_info()
    return {topic: meta.msg_type for topic, meta in info.topics.items()}


def _choose_topic(topic_types: Dict[str, str], requested: str, msg_type: str, preferred: Tuple[str, ...]) -> Optional[str]:
    if requested == "none":
        return None
    if requested != "auto":
        return requested if requested in topic_types else None
    candidates = [topic for topic, typ in topic_types.items() if typ == msg_type]
    for topic in preferred:
        if topic in candidates:
            return topic
    return sorted(candidates)[0] if candidates else None


def _decode_image(msg, image_size: Tuple[int, int]) -> np.ndarray:
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
    resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


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
    occ = np.zeros((grid_size, grid_size), dtype=np.float32)
    height = np.zeros((grid_size, grid_size), dtype=np.float32)
    intensity = np.zeros((grid_size, grid_size), dtype=np.float32)
    count = np.zeros((grid_size, grid_size), dtype=np.float32)

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
        return np.stack([occ, height, intensity], axis=0).astype(np.float16)

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    raw = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    xs = raw["x"].astype(np.float32, copy=False)
    ys = raw["y"].astype(np.float32, copy=False)
    zs = raw["z"].astype(np.float32, copy=False)
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
    ix = ((xs - x_min) * grid_size / max(x_max - x_min, 1e-6)).astype(np.int32)
    iy = ((ys - y_min) * grid_size / max(y_max - y_min, 1e-6)).astype(np.int32)
    ix = np.clip(ix, 0, grid_size - 1)
    iy = np.clip(iy, 0, grid_size - 1)
    rows = grid_size - 1 - ix
    cols = iy
    flat = rows * grid_size + cols

    occ_flat = occ.reshape(-1)
    count_flat = count.reshape(-1)
    height_flat = height.reshape(-1)
    np.add.at(occ_flat, flat, 1.0)
    np.add.at(count_flat, flat, 1.0)
    z_norm = (zs - z_min) / max(z_max - z_min, 1e-6)
    np.maximum.at(height_flat, flat, z_norm)

    if "intensity" in raw.dtype.names:
        inten = raw["intensity"].astype(np.float32, copy=False)[valid]
        intensity_flat = intensity.reshape(-1)
        np.add.at(intensity_flat, flat, inten)

    occ = np.log1p(occ) / np.log(16.0)
    occ = np.clip(occ, 0.0, 1.0)
    nonzero = count > 0
    if "intensity" in raw.dtype.names and np.any(nonzero):
        intensity[nonzero] = intensity[nonzero] / count[nonzero]
        hi = np.percentile(intensity[nonzero], 95)
        if hi > 1e-6:
            intensity = np.clip(intensity / hi, 0.0, 1.0)
    return np.stack([occ, height, intensity], axis=0).astype(np.float16)


def _nearest_indices(source_times: np.ndarray, target_times: np.ndarray, max_gap: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(source_times) == 0:
        return np.zeros(len(target_times), dtype=np.int64), np.zeros(len(target_times), dtype=bool)
    idx = np.searchsorted(source_times, target_times)
    idx = np.clip(idx, 0, len(source_times) - 1)
    prev = np.clip(idx - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(source_times[prev] - target_times) < np.abs(source_times[idx] - target_times)
    nearest = np.where(choose_prev, prev, idx)
    valid = np.abs(source_times[nearest] - target_times) <= max_gap
    return nearest.astype(np.int64), valid


def _nearest_values(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray, max_gap: float) -> Tuple[np.ndarray, np.ndarray]:
    idx, valid = _nearest_indices(source_times, target_times, max_gap)
    out = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    if len(source_times):
        out[valid] = values[idx[valid]]
    return out, valid


def extract_sensor_bag(args: argparse.Namespace) -> None:
    import rosbag

    bag_path = Path(args.bag).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rosbag.Bag(str(bag_path)) as bag:
        topic_types = _topic_table(bag)
        odom_topic = _choose_topic(topic_types, args.odom_topic, "nav_msgs/Odometry", ("/odom", "/lio_localizer/odometry/optimization"))
        imu_topic = _choose_topic(topic_types, args.imu_topic, "sensor_msgs/Imu", ("/imu/data",))
        image_topic = _choose_topic(topic_types, args.image_topic, "sensor_msgs/Image", ("/camera/color/image_raw",))
        lidar_topic = _choose_topic(topic_types, args.lidar_topic, "sensor_msgs/PointCloud2", ("/ouster/points", "/velodyne_points"))
        if odom_topic is None:
            raise RuntimeError("No odometry topic found.")
        if image_topic is None and lidar_topic is None:
            raise RuntimeError("Need at least one exteroceptive sensor topic: Image or PointCloud2.")

        topics = [odom_topic]
        for topic in (imu_topic, image_topic, lidar_topic):
            if topic is not None:
                topics.append(topic)

        odom_rows: List[Tuple[float, float, float, float, float, float]] = []
        imu_rows: List[Tuple[float, float, float, float, float, float, float]] = []
        image_times: List[float] = []
        images: List[np.ndarray] = []
        lidar_times: List[float] = []
        lidar_bevs: List[np.ndarray] = []

        for topic, msg, stamp in bag.read_messages(topics=topics):
            t = stamp.to_sec()
            if topic == odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                tw = msg.twist.twist
                odom_rows.append((t, p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w), tw.linear.x, tw.angular.z))
            elif topic == imu_topic:
                la = msg.linear_acceleration
                av = msg.angular_velocity
                imu_rows.append((t, la.x, la.y, la.z, av.x, av.y, av.z))
            elif topic == image_topic:
                image_times.append(t)
                images.append(_decode_image(msg, tuple(args.image_size)))
            elif topic == lidar_topic:
                lidar_times.append(t)
                lidar_bevs.append(_cloud_to_bev(msg, args.bev_size, args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max))

    odom = np.asarray(odom_rows, dtype=np.float64)
    odom = odom[np.argsort(odom[:, 0])]
    start_time = float(odom[0, 0])
    time = odom[:, 0] - start_time

    if imu_rows:
        imu = np.asarray(imu_rows, dtype=np.float64)
        imu = imu[np.argsort(imu[:, 0])]
        imu_values, imu_valid = _nearest_values(imu[:, 0] - start_time, imu[:, 1:].astype(np.float32), time, args.max_imu_gap)
    else:
        imu_values = np.zeros((len(time), 6), dtype=np.float32)
        imu_valid = np.zeros(len(time), dtype=bool)

    if images:
        image_times_np = np.asarray(image_times, dtype=np.float64) - start_time
        image_order = np.argsort(image_times_np)
        image_times_np = image_times_np[image_order]
        images_np = np.stack([images[i] for i in image_order]).astype(np.uint8)
        image_idx, image_valid = _nearest_indices(image_times_np, time, args.max_image_gap)
        images_by_odom = np.zeros((len(time), images_np.shape[1], images_np.shape[2], 3), dtype=np.uint8)
        images_by_odom[image_valid] = images_np[image_idx[image_valid]]
    else:
        images_by_odom = np.zeros((len(time), args.image_size[1], args.image_size[0], 3), dtype=np.uint8)
        image_valid = np.zeros(len(time), dtype=bool)

    if lidar_bevs:
        lidar_times_np = np.asarray(lidar_times, dtype=np.float64) - start_time
        lidar_order = np.argsort(lidar_times_np)
        lidar_times_np = lidar_times_np[lidar_order]
        lidar_np = np.stack([lidar_bevs[i] for i in lidar_order]).astype(np.float16)
        lidar_idx, lidar_valid = _nearest_indices(lidar_times_np, time, args.max_lidar_gap)
        lidar_by_odom = np.zeros((len(time), lidar_np.shape[1], lidar_np.shape[2], lidar_np.shape[3]), dtype=np.float16)
        lidar_by_odom[lidar_valid] = lidar_np[lidar_idx[lidar_valid]]
    else:
        lidar_by_odom = np.zeros((len(time), 3, args.bev_size, args.bev_size), dtype=np.float16)
        lidar_valid = np.zeros(len(time), dtype=bool)

    meta = {
        "bag": str(bag_path),
        "odom_topic": odom_topic,
        "imu_topic": imu_topic,
        "image_topic": image_topic,
        "lidar_topic": lidar_topic,
        "image_size_wh": args.image_size,
        "bev_size": args.bev_size,
        "bev_bounds": {
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
            "z_min": args.z_min,
            "z_max": args.z_max,
        },
        "notes": "cmd_vel is intentionally not used as the policy target.",
    }

    np.savez_compressed(
        out_path,
        time=time.astype(np.float32),
        x=odom[:, 1].astype(np.float32),
        y=odom[:, 2].astype(np.float32),
        yaw=odom[:, 3].astype(np.float32),
        v=odom[:, 4].astype(np.float32),
        w=odom[:, 5].astype(np.float32),
        imu=imu_values.astype(np.float32),
        imu_valid=imu_valid,
        images=images_by_odom,
        image_valid=image_valid,
        lidar_bev=lidar_by_odom,
        lidar_valid=lidar_valid,
        meta=json.dumps(meta, ensure_ascii=False),
    )

    print(json.dumps({
        "output": str(out_path),
        "odom_messages": int(len(time)),
        "image_topic": image_topic,
        "image_coverage": float(np.mean(image_valid)),
        "lidar_topic": lidar_topic,
        "lidar_coverage": float(np.mean(lidar_valid)),
        "imu_coverage": float(np.mean(imu_valid)),
        "duration_sec": float(time[-1] - time[0]),
    }, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract odom-aligned camera/LiDAR route memory from a ROS1 bag.")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--odom-topic", default="auto")
    parser.add_argument("--imu-topic", default="auto")
    parser.add_argument("--image-topic", default="auto")
    parser.add_argument("--lidar-topic", default="auto")
    parser.add_argument("--image-size", type=int, nargs=2, default=[160, 96], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--bev-size", type=int, default=64)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=12.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--max-image-gap", type=float, default=0.12)
    parser.add_argument("--max-lidar-gap", type=float, default=0.18)
    parser.add_argument("--max-imu-gap", type=float, default=0.05)
    return parser


def main() -> None:
    extract_sensor_bag(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
