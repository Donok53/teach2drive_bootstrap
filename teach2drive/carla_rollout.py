import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np
import torch

from .carla_collect import _carla_image_to_rgb, _carla_lidar_to_bev, _destroy_actors, _get_matching, _import_carla
from .geometry import cumulative_distance, pose_to_ego, wrap_angle
from .model import SensorFusionPolicy


def _load_model(checkpoint, device):
    ckpt = torch.load(str(checkpoint), map_location=device)
    cfg = ckpt["model"]
    model = SensorFusionPolicy(
        scalar_dim=int(cfg["scalar_dim"]),
        output_dim=int(cfg["output_dim"]),
        embed_dim=int(cfg["embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {
        "scalar_mean": ckpt["scalar_mean"].astype(np.float32),
        "scalar_std": ckpt["scalar_std"].astype(np.float32),
        "target_mean": ckpt["target_mean"].astype(np.float32),
        "target_std": ckpt["target_std"].astype(np.float32),
    }
    return model, norm


def _load_route(route_npz):
    data = np.load(str(route_npz), allow_pickle=False)
    if "route" in data.files:
        route = data["route"].astype(np.float64)
        if route.shape[1] >= 4:
            return route[:, :4], json.loads(data["meta"].item()) if "meta" in data.files else {}
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    yaw = data["yaw"].astype(np.float64)
    s = cumulative_distance(x, y)
    return np.stack([x, y, yaw, s], axis=1), json.loads(data["meta"].item()) if "meta" in data.files else {}


def _nearest_route(route, x, y, yaw, heading_score_weight):
    d = np.linalg.norm(route[:, :2] - np.asarray([x, y]), axis=1)
    yaw_err = np.abs(wrap_angle(route[:, 2] - yaw))
    idx = int(np.argmin(d + heading_score_weight * yaw_err))
    return idx, float(d[idx])


def _projected_speed(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    yaw = math.radians(transform.rotation.yaw)
    return float(velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw))


def _short_map_name(map_name):
    return str(map_name).rsplit("/", 1)[-1] if map_name else ""


def _make_scalar(route, route_len, odom, imu, image_valid, lidar_valid, lookahead_m, heading_score_weight):
    x, y, yaw, v, w = [float(value) for value in odom]
    nearest_idx, route_dist = _nearest_route(route, x, y, yaw, heading_score_weight)
    lookahead_idx = int(np.searchsorted(route[:, 3], route[nearest_idx, 3] + lookahead_m))
    lookahead_idx = min(lookahead_idx, len(route) - 1)
    current_pose = (x, y, yaw)
    anchor_pose = tuple(float(v) for v in route[nearest_idx, :3])
    lookahead_pose = tuple(float(v) for v in route[lookahead_idx, :3])
    gx, gy, gyaw = pose_to_ego(*current_pose, *lookahead_pose)
    ax, ay, ayaw = pose_to_ego(*current_pose, *anchor_pose)
    scalar = np.asarray(
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
            float(route[nearest_idx, 3] / max(route_len, 1e-6)),
            float(route_len - route[nearest_idx, 3]),
        ],
        dtype=np.float32,
    )
    return scalar, nearest_idx, route_dist


def _predict(model, norm, device, scalar, image, lidar):
    scalar_norm = (scalar - norm["scalar_mean"]) / norm["scalar_std"]
    scalar_t = torch.from_numpy(scalar_norm).float().unsqueeze(0).to(device)
    image_t = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar.astype(np.float32)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(scalar_t, image_t, lidar_t).cpu().numpy()[0]
    pred = pred_norm * norm["target_std"] + norm["target_mean"]
    return pred.reshape(-1, 3)


def _apply_control(carla, vehicle, pred, args):
    target = pred[min(args.control_point_index, len(pred) - 1)]
    x, y, _yaw = [float(v) for v in target]
    dist = max(math.hypot(x, y), 1e-3)
    curvature = 2.0 * y / max(dist * dist, 1e-3)
    steer_angle = math.atan(args.wheelbase_m * curvature)
    steer = float(np.clip(steer_angle / max(args.max_steer_rad, 1e-3), -1.0, 1.0))
    desired_speed = float(np.clip(dist / max(args.control_horizon_sec, 1e-3), args.min_speed, args.max_speed))
    current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
    speed_error = desired_speed - current_speed
    throttle = float(np.clip(args.speed_kp * speed_error, 0.0, args.max_throttle))
    brake = float(np.clip(-args.brake_kp * speed_error, 0.0, args.max_brake))
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))


def rollout(args):
    carla = _import_carla()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, norm = _load_model(Path(args.checkpoint).expanduser(), device)
    route, meta = _load_route(Path(args.route_npz).expanduser())
    route_len = float(route[-1, 3])

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    map_name = args.map or meta.get("map", "")
    if map_name:
        requested_map = _short_map_name(map_name)
        current_map = _short_map_name(world.get_map().name)
        if requested_map and requested_map != current_map:
            world = client.load_world(requested_map)

    original_settings = world.get_settings()
    actors = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.filter(args.vehicle_filter)[0]
        start = route[args.start_index]
        spawn = carla.Transform(
            carla.Location(x=float(start[0]), y=float(start[1]), z=args.spawn_z),
            carla.Rotation(yaw=math.degrees(float(start[2]))),
        )
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)

        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(args.image_size[0]))
        camera_bp.set_attribute("image_size_y", str(args.image_size[1]))
        camera_bp.set_attribute("fov", str(args.camera_fov))
        camera_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=-8.0)),
            attach_to=vehicle,
        )
        actors.append(camera)

        lidar_bp = blueprints.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(args.lidar_channels))
        lidar_bp.set_attribute("range", str(args.lidar_range))
        lidar_bp.set_attribute("points_per_second", str(args.lidar_points_per_second))
        lidar_bp.set_attribute("rotation_frequency", str(args.hz))
        lidar_bp.set_attribute("upper_fov", str(args.lidar_upper_fov))
        lidar_bp.set_attribute("lower_fov", str(args.lidar_lower_fov))
        lidar_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
        lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(z=1.8)), attach_to=vehicle)
        actors.append(lidar)

        imu_bp = blueprints.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
        imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        actors.append(imu)

        camera_q = queue.Queue()
        lidar_q = queue.Queue()
        imu_q = queue.Queue()
        camera.listen(camera_q.put)
        lidar.listen(lidar_q.put)
        imu.listen(imu_q.put)

        for _ in range(max(int(args.warmup_sec * args.hz), 0)):
            world.tick()

        max_steps = int(args.duration_sec * args.hz)
        cross_track_errors = []
        progress_values = []
        success = False
        for step in range(max_steps):
            frame = world.tick()
            camera_data = _get_matching(camera_q, frame)
            lidar_data = _get_matching(lidar_q, frame)
            imu_data = _get_matching(imu_q, frame)

            transform = vehicle.get_transform()
            location = transform.location
            yaw = math.radians(transform.rotation.yaw)
            angular_velocity = vehicle.get_angular_velocity()
            odom = np.asarray([location.x, location.y, yaw, _projected_speed(vehicle), math.radians(float(angular_velocity.z))], dtype=np.float32)
            imu_values = np.asarray(
                [
                    imu_data.accelerometer.x,
                    imu_data.accelerometer.y,
                    imu_data.accelerometer.z,
                    imu_data.gyroscope.x,
                    imu_data.gyroscope.y,
                    imu_data.gyroscope.z,
                ],
                dtype=np.float32,
            )
            image = _carla_image_to_rgb(camera_data, args.image_size)
            lidar_bev = _carla_lidar_to_bev(lidar_data, args)
            scalar, nearest_idx, route_dist = _make_scalar(route, route_len, odom, imu_values, True, True, args.lookahead_m, args.heading_score_weight)
            pred = _predict(model, norm, device, scalar, image, lidar_bev)
            _apply_control(carla, vehicle, pred, args)

            cross_track_errors.append(route_dist)
            progress_values.append(float(route[nearest_idx, 3]))
            if route_len - route[nearest_idx, 3] <= args.goal_tolerance_m:
                success = True
                break
            if route_dist > args.failure_distance_m:
                break

        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        metrics = {
            "success": success,
            "steps": len(cross_track_errors),
            "route_length_m": route_len,
            "final_progress_m": progress_values[-1] if progress_values else 0.0,
            "mean_cross_track_error_m": float(np.mean(cross_track_errors)) if cross_track_errors else None,
            "max_cross_track_error_m": float(np.max(cross_track_errors)) if cross_track_errors else None,
            "device": str(device),
        }
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    finally:
        _destroy_actors(client, carla, actors)
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Closed-loop CARLA rollout for a Teach2Drive sensor model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--spawn-z", type=float, default=0.5)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--image-size", type=int, nargs=2, default=[160, 96], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--bev-size", type=int, default=64)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=12.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=10.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--lidar-channels", type=int, default=32)
    parser.add_argument("--lidar-range", type=float, default=50.0)
    parser.add_argument("--lidar-points-per-second", type=int, default=120000)
    parser.add_argument("--lidar-upper-fov", type=float, default=10.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-25.0)
    parser.add_argument("--lookahead-m", type=float, default=2.0)
    parser.add_argument("--heading-score-weight", type=float, default=0.2)
    parser.add_argument("--control-point-index", type=int, default=1)
    parser.add_argument("--control-horizon-sec", type=float, default=1.0)
    parser.add_argument("--wheelbase-m", type=float, default=2.8)
    parser.add_argument("--max-steer-rad", type=float, default=0.6)
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--brake-kp", type=float, default=0.5)
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.7)
    parser.add_argument("--goal-tolerance-m", type=float, default=5.0)
    parser.add_argument("--failure-distance-m", type=float, default=8.0)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    rollout(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
