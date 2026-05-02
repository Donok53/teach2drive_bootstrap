import math
from typing import Iterable, Tuple

import numpy as np


def wrap_angle(angle):
    """Wrap radians to [-pi, pi]. Works with scalars or numpy arrays."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_to_ego(
    current_x: float,
    current_y: float,
    current_yaw: float,
    target_x: float,
    target_y: float,
    target_yaw: float,
) -> Tuple[float, float, float]:
    dx = target_x - current_x
    dy = target_y - current_y
    c = math.cos(current_yaw)
    s = math.sin(current_yaw)
    ego_x = c * dx + s * dy
    ego_y = -s * dx + c * dy
    ego_yaw = float(wrap_angle(target_yaw - current_yaw))
    return ego_x, ego_y, ego_yaw


def batch_pose_to_ego(current_pose: np.ndarray, target_poses: np.ndarray) -> np.ndarray:
    """Transform target poses [N, 3] into the ego frame of current_pose [3]."""
    cx, cy, cyaw = current_pose
    dx = target_poses[:, 0] - cx
    dy = target_poses[:, 1] - cy
    c = math.cos(cyaw)
    s = math.sin(cyaw)
    ego_x = c * dx + s * dy
    ego_y = -s * dx + c * dy
    ego_yaw = wrap_angle(target_poses[:, 2] - cyaw)
    return np.stack([ego_x, ego_y, ego_yaw], axis=1)


def cumulative_distance(xs: Iterable[float], ys: Iterable[float]) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) == 0:
        return np.zeros(0, dtype=np.float64)
    dist = np.zeros(len(xs), dtype=np.float64)
    if len(xs) > 1:
        step = np.hypot(np.diff(xs), np.diff(ys))
        dist[1:] = np.cumsum(step)
    return dist


def perturb_pose(
    x: float,
    y: float,
    yaw: float,
    lateral_m: float,
    forward_m: float,
    yaw_rad: float,
) -> Tuple[float, float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    px = x + forward_m * c - lateral_m * s
    py = y + forward_m * s + lateral_m * c
    pyaw = float(wrap_angle(yaw + yaw_rad))
    return px, py, pyaw

