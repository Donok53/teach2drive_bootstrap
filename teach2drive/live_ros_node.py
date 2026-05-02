import argparse
import math
import threading
from pathlib import Path as FilePath
from typing import Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from sensor_msgs.msg import Image, Imu, PointCloud2

from .geometry import cumulative_distance, pose_to_ego, quat_to_yaw, wrap_angle
from .model import SensorFusionPolicy
from .sensor_extract import _cloud_to_bev, _decode_image


class Teach2DriveNode:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.latest_odom = None
        self.latest_odom_stamp = None
        self.latest_imu = np.zeros(6, dtype=np.float32)
        self.latest_imu_stamp = None
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_lidar = None
        self.latest_lidar_stamp = None

        self.route = self._load_route(FilePath(args.route_npz).expanduser())
        self.route_xy = self.route[:, :2]
        self.route_yaw = self.route[:, 2]
        self.route_s = self.route[:, 3]
        self.route_len = float(self.route_s[-1])

        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.model, self.norm = self._load_model(FilePath(args.checkpoint).expanduser())
        self.model.eval()

        self.path_pub = rospy.Publisher(args.predicted_path_topic, Path, queue_size=1)
        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist, queue_size=1) if args.publish_cmd_vel else None

        rospy.Subscriber(args.odom_topic, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(args.imu_topic, Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber(args.image_topic, Image, self._image_cb, queue_size=1)
        rospy.Subscriber(args.lidar_topic, PointCloud2, self._lidar_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / args.rate), self._timer_cb)

        rospy.loginfo(
            "Teach2Drive live node ready: device=%s route_len=%.2fm publish_cmd_vel=%s",
            self.device,
            self.route_len,
            args.publish_cmd_vel,
        )

    def _load_model(self, checkpoint_path: FilePath):
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        cfg = ckpt["model"]
        model = SensorFusionPolicy(
            scalar_dim=int(cfg["scalar_dim"]),
            output_dim=int(cfg["output_dim"]),
            embed_dim=int(cfg["embed_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg.get("dropout", 0.0)),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        norm = {
            "scalar_mean": ckpt["scalar_mean"].astype(np.float32),
            "scalar_std": ckpt["scalar_std"].astype(np.float32),
            "target_mean": ckpt["target_mean"].astype(np.float32),
            "target_std": ckpt["target_std"].astype(np.float32),
        }
        return model, norm

    def _load_route(self, route_path: FilePath) -> np.ndarray:
        data = np.load(str(route_path), allow_pickle=False)
        if "route" in data.files:
            route = data["route"].astype(np.float64)
            if route.shape[1] >= 4:
                return route[:, :4]
        x = data["x"].astype(np.float64)
        y = data["y"].astype(np.float64)
        yaw = data["yaw"].astype(np.float64)
        s = cumulative_distance(x, y)
        return np.stack([x, y, yaw, s], axis=1)

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        tw = msg.twist.twist
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            self.latest_odom = np.asarray(
                [p.x, p.y, quat_to_yaw(q.x, q.y, q.z, q.w), tw.linear.x, tw.angular.z],
                dtype=np.float32,
            )
            self.latest_odom_stamp = stamp

    def _imu_cb(self, msg: Imu):
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            self.latest_imu = np.asarray(
                [
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z,
                ],
                dtype=np.float32,
            )
            self.latest_imu_stamp = stamp

    def _image_cb(self, msg: Image):
        try:
            image = _decode_image(msg, tuple(self.args.image_size))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image decode failed: %s", exc)
            return
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = stamp

    def _lidar_cb(self, msg: PointCloud2):
        try:
            bev = _cloud_to_bev(
                msg,
                self.args.bev_size,
                self.args.x_min,
                self.args.x_max,
                self.args.y_min,
                self.args.y_max,
                self.args.z_min,
                self.args.z_max,
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "lidar BEV conversion failed: %s", exc)
            return
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()
        with self.lock:
            self.latest_lidar = bev
            self.latest_lidar_stamp = stamp

    def _nearest_route_index(self, x: float, y: float, yaw: float) -> Tuple[int, float]:
        d = np.linalg.norm(self.route_xy - np.asarray([x, y]), axis=1)
        yaw_err = np.abs(wrap_angle(self.route_yaw - yaw))
        score = d + self.args.heading_score_weight * yaw_err
        idx = int(np.argmin(score))
        return idx, float(d[idx])

    def _make_scalar_feature(self, odom, imu, image_valid, lidar_valid):
        x, y, yaw, v, w = [float(value) for value in odom]
        nearest_idx, route_dist = self._nearest_route_index(x, y, yaw)
        if route_dist > self.args.max_route_distance:
            return None, None, None

        lookahead_s = self.route_s[nearest_idx] + self.args.lookahead_m
        lookahead_idx = int(np.searchsorted(self.route_s, lookahead_s))
        lookahead_idx = min(lookahead_idx, len(self.route) - 1)

        current_pose = (x, y, yaw)
        anchor_pose = tuple(float(v) for v in self.route[nearest_idx, :3])
        lookahead_pose = tuple(float(v) for v in self.route[lookahead_idx, :3])
        gx, gy, gyaw = pose_to_ego(*current_pose, *lookahead_pose)
        ax, ay, ayaw = pose_to_ego(*current_pose, *anchor_pose)
        feature = np.asarray(
            [
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
                float(self.route_s[nearest_idx] / max(self.route_len, 1e-6)),
                float(self.route_len - self.route_s[nearest_idx]),
            ],
            dtype=np.float32,
        )
        return feature, nearest_idx, lookahead_idx

    def _timer_cb(self, _event):
        now = rospy.Time.now().to_sec()
        with self.lock:
            odom = None if self.latest_odom is None else self.latest_odom.copy()
            odom_stamp = self.latest_odom_stamp
            imu = self.latest_imu.copy()
            imu_stamp = self.latest_imu_stamp
            image = None if self.latest_image is None else self.latest_image.copy()
            image_stamp = self.latest_image_stamp
            lidar = None if self.latest_lidar is None else self.latest_lidar.copy()
            lidar_stamp = self.latest_lidar_stamp

        if odom is None:
            return
        if odom_stamp is not None and now - odom_stamp > self.args.max_odom_age:
            self._publish_stop("stale odom")
            return

        image_valid = image is not None and image_stamp is not None and now - image_stamp <= self.args.max_sensor_age
        lidar_valid = lidar is not None and lidar_stamp is not None and now - lidar_stamp <= self.args.max_sensor_age
        imu_valid = imu_stamp is not None and now - imu_stamp <= self.args.max_sensor_age
        if not image_valid:
            image = np.zeros((self.args.image_size[1], self.args.image_size[0], 3), dtype=np.uint8)
        if not lidar_valid:
            lidar = np.zeros((3, self.args.bev_size, self.args.bev_size), dtype=np.float16)
        if not imu_valid:
            imu = np.zeros(6, dtype=np.float32)

        if not image_valid and not lidar_valid:
            self._publish_stop("no fresh camera/lidar")
            return

        scalar, nearest_idx, lookahead_idx = self._make_scalar_feature(odom, imu, image_valid, lidar_valid)
        if scalar is None:
            self._publish_stop("too far from route")
            return

        pred = self._predict(scalar, image, lidar)
        self._publish_path(pred, odom)
        if self.cmd_pub is not None:
            self._publish_cmd(pred)

    def _predict(self, scalar, image, lidar):
        scalar_norm = (scalar - self.norm["scalar_mean"]) / self.norm["scalar_std"]
        scalar_t = torch.from_numpy(scalar_norm).float().unsqueeze(0).to(self.device)
        image_t = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(self.device)
        lidar_t = torch.from_numpy(lidar.astype(np.float32)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model(scalar_t, image_t, lidar_t).cpu().numpy()[0]
        pred = pred_norm * self.norm["target_std"] + self.norm["target_mean"]
        return pred.reshape(-1, 3)

    def _publish_path(self, pred, odom):
        path = PathMsg()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.args.base_frame
        for dx, dy, dyaw in pred:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(dx)
            pose.pose.position.y = float(dy)
            pose.pose.position.z = 0.0
            yaw = float(dyaw)
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_cmd(self, pred):
        target = pred[min(self.args.control_point_index, len(pred) - 1)]
        x, y, _yaw = [float(v) for v in target]
        distance = max(math.hypot(x, y), 1e-3)
        curvature = 2.0 * y / max(distance * distance, 1e-3)
        speed = min(self.args.max_speed, max(self.args.min_speed, distance / max(self.args.control_horizon_sec, 1e-3)))
        omega = float(np.clip(speed * curvature, -self.args.max_yaw_rate, self.args.max_yaw_rate))
        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

    def _publish_stop(self, reason):
        rospy.logwarn_throttle(2.0, "Teach2Drive stop: %s", reason)
        if self.cmd_pub is not None:
            self.cmd_pub.publish(Twist())


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run Teach2Drive sensor policy as a ROS1 live node.")
    parser.add_argument("--checkpoint", default="runs/camera_overlap/best_sensor_model.pt")
    parser.add_argument("--route-npz", default="runs/camera_overlap/sensor_route.npz")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--lidar-topic", default="/ouster/points")
    parser.add_argument("--predicted-path-topic", default="/teach2drive/predicted_path")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--lookahead-m", type=float, default=2.0)
    parser.add_argument("--max-route-distance", type=float, default=2.0)
    parser.add_argument("--heading-score-weight", type=float, default=0.2)
    parser.add_argument("--max-odom-age", type=float, default=0.5)
    parser.add_argument("--max-sensor-age", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, nargs=2, default=[160, 96], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--bev-size", type=int, default=64)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=12.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--publish-cmd-vel", action="store_true")
    parser.add_argument("--control-point-index", type=int, default=1)
    parser.add_argument("--control-horizon-sec", type=float, default=1.0)
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=0.25)
    parser.add_argument("--max-yaw-rate", type=float, default=0.6)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    rospy.init_node("teach2drive_live_node")
    Teach2DriveNode(args)
    rospy.spin()


if __name__ == "__main__":
    main()
