import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import quat_to_yaw


def _topic_table(bag) -> Dict[str, str]:
    info = bag.get_type_and_topic_info()
    return {topic: meta.msg_type for topic, meta in info.topics.items()}


def _choose_topic(topic_types: Dict[str, str], requested: str, msg_type: str) -> Optional[str]:
    if requested != "auto":
        return requested if requested in topic_types else None
    candidates = [topic for topic, typ in topic_types.items() if typ == msg_type]
    if not candidates:
        return None
    preferred = ["/odom", "/lio_localizer/odometry/optimization", "/imu/data"]
    for topic in preferred:
        if topic in candidates:
            return topic
    return sorted(candidates)[0]


def _nearest_resample(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray, max_gap: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(source_times) == 0:
        return np.zeros((len(target_times), values.shape[1] if values.ndim == 2 else 0), dtype=np.float32), np.zeros(len(target_times), dtype=bool)
    indices = np.searchsorted(source_times, target_times)
    indices = np.clip(indices, 0, len(source_times) - 1)
    prev_indices = np.clip(indices - 1, 0, len(source_times) - 1)
    choose_prev = np.abs(source_times[prev_indices] - target_times) < np.abs(source_times[indices] - target_times)
    nearest = np.where(choose_prev, prev_indices, indices)
    gap = np.abs(source_times[nearest] - target_times)
    valid = gap <= max_gap
    out = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    out[valid] = values[nearest[valid]]
    return out, valid


def extract_bag(args: argparse.Namespace) -> None:
    import rosbag

    bag_path = Path(args.bag).expanduser()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rosbag.Bag(str(bag_path)) as bag:
        topic_types = _topic_table(bag)
        odom_topic = _choose_topic(topic_types, args.odom_topic, "nav_msgs/Odometry")
        imu_topic = _choose_topic(topic_types, args.imu_topic, "sensor_msgs/Imu")

        if odom_topic is None:
            raise RuntimeError("No odometry topic found. Pass --odom-topic explicitly.")

        odom_rows: List[Tuple[float, float, float, float, float, float]] = []
        imu_rows: List[Tuple[float, float, float, float, float, float, float]] = []
        read_topics = [odom_topic]
        if imu_topic is not None:
            read_topics.append(imu_topic)

        for topic, msg, stamp in bag.read_messages(topics=read_topics):
            t = stamp.to_sec()
            if topic == odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                twist = msg.twist.twist
                yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
                odom_rows.append((t, p.x, p.y, yaw, twist.linear.x, twist.angular.z))
            elif topic == imu_topic:
                av = msg.angular_velocity
                la = msg.linear_acceleration
                imu_rows.append((t, la.x, la.y, la.z, av.x, av.y, av.z))

    if len(odom_rows) < 2:
        raise RuntimeError(f"Odometry topic {odom_topic} produced too few messages: {len(odom_rows)}")

    odom = np.asarray(odom_rows, dtype=np.float64)
    order = np.argsort(odom[:, 0])
    odom = odom[order]
    times = odom[:, 0]
    start_time = float(times[0])
    rel_times = times - start_time

    if imu_rows:
        imu = np.asarray(imu_rows, dtype=np.float64)
        imu = imu[np.argsort(imu[:, 0])]
        imu_values, imu_valid = _nearest_resample(
            imu[:, 0] - start_time,
            imu[:, 1:].astype(np.float32),
            rel_times,
            args.max_imu_gap,
        )
    else:
        imu_values = np.zeros((len(rel_times), 6), dtype=np.float32)
        imu_valid = np.zeros(len(rel_times), dtype=bool)

    sensor_topics = {
        topic: typ
        for topic, typ in sorted(topic_types.items())
        if typ.startswith("sensor_msgs/") or typ in {"nav_msgs/Odometry", "geometry_msgs/Twist"}
    }
    meta = {
        "bag": str(bag_path),
        "odom_topic": odom_topic,
        "imu_topic": imu_topic,
        "start_time": start_time,
        "sensor_topics": sensor_topics,
        "notes": "cmd_vel is recorded in some bags but is intentionally not used as the policy target.",
    }

    np.savez_compressed(
        out_path,
        time=rel_times.astype(np.float32),
        x=odom[:, 1].astype(np.float32),
        y=odom[:, 2].astype(np.float32),
        yaw=odom[:, 3].astype(np.float32),
        v=odom[:, 4].astype(np.float32),
        w=odom[:, 5].astype(np.float32),
        imu=imu_values.astype(np.float32),
        imu_valid=imu_valid,
        meta=json.dumps(meta, ensure_ascii=False),
    )

    print(json.dumps({
        "output": str(out_path),
        "odom_topic": odom_topic,
        "imu_topic": imu_topic,
        "odom_messages": int(len(rel_times)),
        "imu_coverage": float(np.mean(imu_valid)) if len(imu_valid) else 0.0,
        "duration_sec": float(rel_times[-1] - rel_times[0]),
    }, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract route memory arrays from a ROS1 bag.")
    parser.add_argument("--bag", required=True, help="Path to a ROS1 .bag file.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--odom-topic", default="auto", help="Odometry topic or 'auto'.")
    parser.add_argument("--imu-topic", default="auto", help="IMU topic, 'auto', or a missing topic to disable.")
    parser.add_argument("--max-imu-gap", type=float, default=0.05, help="Max seconds for nearest IMU sample.")
    return parser


def main() -> None:
    extract_bag(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

