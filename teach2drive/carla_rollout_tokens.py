import argparse
import json
import math
import queue
from pathlib import Path

import cv2
import numpy as np
import torch

from .carla_collect import _carla_image_to_rgb, _carla_lidar_to_bev, _destroy_actors, _get_matching, _import_carla
from .carla_collect_tokens import CAMERA_TRANSFORMS
from .carla_rollout import (
    _camera_blueprint,
    _collision_key,
    _location_text,
    _new_infraction_log,
    _open_video_writer,
    _render_video_frame,
    _score_like_leaderboard,
    _short_map_name,
)
from .geometry import cumulative_distance, pose_to_ego, wrap_angle
from .model import TokenFusionPolicy


def _load_token_model(checkpoint, device):
    ckpt = torch.load(str(checkpoint), map_location=device)
    cfg = ckpt["model"]
    model = TokenFusionPolicy(
        scalar_dim=int(cfg["scalar_dim"]),
        output_dim=int(cfg["output_dim"]),
        num_cameras=int(cfg["num_cameras"]),
        embed_dim=int(cfg["embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        transformer_layers=int(cfg["transformer_layers"]),
        num_heads=int(cfg["num_heads"]),
        dropout=float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {
        "scalar_mean": ckpt["scalar_mean"].astype(np.float32),
        "scalar_std": ckpt["scalar_std"].astype(np.float32),
        "target_mean": ckpt["target_mean"].astype(np.float32),
        "target_std": ckpt["target_std"].astype(np.float32),
        "traj_dim": int(ckpt["traj_dim"]),
        "speed_dim": int(ckpt["speed_dim"]),
        "cameras": list(ckpt.get("cameras", ["front", "left", "right"])),
    }
    return model, norm


def _read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda item: (int(item.get("step", 0)), float(item.get("time", 0.0))))
    return records


def _load_token_route(route_source, episode_index):
    source = Path(route_source).expanduser()
    if source.is_file() and source.suffix == ".npz":
        arrays = np.load(source, allow_pickle=False)
        episode_dir = Path(str(arrays["episode_dirs"][episode_index]))
    else:
        episode_dir = source
    frames_path = episode_dir / "frames.jsonl"
    if not frames_path.exists():
        raise FileNotFoundError(f"Could not find token episode frames: {frames_path}")
    frames = _read_jsonl(frames_path)
    x = np.asarray([float(frame["odom"]["x"]) for frame in frames], dtype=np.float64)
    y = np.asarray([float(frame["odom"]["y"]) for frame in frames], dtype=np.float64)
    yaw = np.asarray([float(frame["odom"]["yaw"]) for frame in frames], dtype=np.float64)
    route_s = cumulative_distance(x, y)
    route = np.stack([x, y, yaw, route_s], axis=1)
    meta_path = episode_dir / "episode_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return route, meta, episode_dir


def _camera_transform(carla, name):
    location, rotation = CAMERA_TRANSFORMS[name]
    return carla.Transform(
        carla.Location(x=location[0], y=location[1], z=location[2]),
        carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
    )


def _spawn_cameras(carla, world, blueprints, vehicle, cameras, args, actors):
    sensors = {}
    queues = {}
    for name in cameras:
        camera_bp = _camera_blueprint(blueprints, args.image_size, args.camera_fov, args.hz)
        sensor = world.spawn_actor(camera_bp, _camera_transform(carla, name), attach_to=vehicle)
        actors.append(sensor)
        sensor_q = queue.Queue()
        sensor.listen(sensor_q.put)
        sensors[name] = sensor
        queues[name] = sensor_q
    return sensors, queues


def _spawn_lidar(carla, world, blueprints, vehicle, args, actors):
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
    lidar_q = queue.Queue()
    lidar.listen(lidar_q.put)
    return lidar, lidar_q


def _spawn_imu(carla, world, blueprints, vehicle, args, actors):
    imu_bp = blueprints.find("sensor.other.imu")
    imu_bp.set_attribute("sensor_tick", str(1.0 / args.hz))
    imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
    actors.append(imu)
    imu_q = queue.Queue()
    imu.listen(imu_q.put)
    return imu, imu_q


def _projected_speed(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    yaw = math.radians(transform.rotation.yaw)
    return float(velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw))


def _nearest_route_monotonic(route, x, y, yaw, previous_idx, args):
    if previous_idx is None:
        lo = 0
        hi = len(route)
    else:
        lo = max(int(previous_idx) - args.route_search_back, 0)
        hi = min(int(previous_idx) + args.route_search_ahead + 1, len(route))
    candidate = route[lo:hi]
    d = np.linalg.norm(candidate[:, :2] - np.asarray([x, y]), axis=1)
    yaw_err = np.abs(wrap_angle(candidate[:, 2] - yaw))
    local_idx = int(np.argmin(d + args.heading_score_weight * yaw_err))
    nearest_idx = lo + local_idx
    if previous_idx is not None:
        nearest_idx = max(nearest_idx, max(int(previous_idx) - args.route_search_back, 0))
    return nearest_idx, float(d[local_idx])


def _make_scalar_monotonic(route, route_len, odom, imu, image_valid, lidar_valid, previous_idx, args):
    x, y, yaw, v, w = [float(value) for value in odom]
    nearest_idx, route_dist = _nearest_route_monotonic(route, x, y, yaw, previous_idx, args)
    lookahead_idx = int(np.searchsorted(route[:, 3], route[nearest_idx, 3] + args.lookahead_m))
    lookahead_idx = min(lookahead_idx, len(route) - 1)
    current_pose = (x, y, yaw)
    anchor_pose = tuple(float(value) for value in route[nearest_idx, :3])
    lookahead_pose = tuple(float(value) for value in route[lookahead_idx, :3])
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


def _predict_token(model, norm, device, scalar, images, lidar):
    scalar_norm = (scalar - norm["scalar_mean"]) / norm["scalar_std"]
    target_dim = int(norm["traj_dim"] + norm["speed_dim"])
    scalar_t = torch.from_numpy(scalar_norm).float().unsqueeze(0).to(device)
    cameras_t = torch.from_numpy(images.astype(np.float32).transpose(0, 3, 1, 2) / 255.0).float().unsqueeze(0).to(device)
    lidar_t = torch.from_numpy(lidar.astype(np.float32)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred_raw = model(scalar_t, cameras_t, lidar_t).cpu().numpy()[0]
    target = pred_raw[:target_dim] * norm["target_std"] + norm["target_mean"]
    traj = target[: norm["traj_dim"]].reshape(-1, 3)
    speeds = target[norm["traj_dim"] : target_dim]
    stop_prob = float(1.0 / (1.0 + np.exp(-pred_raw[target_dim])))
    return traj, speeds, stop_prob


def _apply_control(carla, vehicle, traj, speeds, stop_prob, args, control_state):
    current_speed = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y)
    if args.use_stop_head and stop_prob >= args.stop_prob_threshold:
        control_state["desired_speed"] = 0.0
        control_state["steer"] = control_state.get("steer", 0.0) * args.steer_smoothing
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=float(control_state["steer"]), brake=args.max_brake))
        return
    target_idx = min(args.control_point_index, len(traj) - 1)
    target = traj[target_idx]
    x, y, _yaw = [float(value) for value in target]
    dist = max(math.hypot(x, y), 1e-3)
    curvature = 2.0 * y / max(dist * dist, 1e-3)
    steer_angle = math.atan(args.wheelbase_m * curvature)
    steer = float(np.clip(steer_angle / max(args.max_steer_rad, 1e-3), -1.0, 1.0))
    geometry_speed = dist / max(args.control_horizon_sec, 1e-3)
    head_speed = abs(float(speeds[min(target_idx, len(speeds) - 1)])) if len(speeds) else geometry_speed
    desired_speed = args.speed_head_mix * head_speed + (1.0 - args.speed_head_mix) * geometry_speed
    desired_speed = float(np.clip(desired_speed, args.min_speed, args.max_speed))

    previous_speed = float(control_state.get("desired_speed", current_speed))
    max_up = args.max_accel_mps2 / max(args.hz, 1)
    max_down = args.max_decel_mps2 / max(args.hz, 1)
    desired_speed = float(np.clip(desired_speed, previous_speed - max_down, previous_speed + max_up))
    control_state["desired_speed"] = desired_speed
    previous_steer = float(control_state.get("steer", steer))
    steer = args.steer_smoothing * previous_steer + (1.0 - args.steer_smoothing) * steer
    control_state["steer"] = steer

    speed_error = desired_speed - current_speed
    throttle = float(np.clip(args.speed_kp * speed_error, 0.0, args.max_throttle))
    brake = float(np.clip(-args.brake_kp * speed_error, 0.0, args.max_brake))
    vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))


def rollout(args):
    carla = _import_carla()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, norm = _load_token_model(Path(args.checkpoint).expanduser(), device)
    route, meta, episode_dir = _load_token_route(args.route_source, args.episode_index)
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
    video_writer = None
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

        cameras = norm["cameras"]
        _camera_sensors, camera_queues = _spawn_cameras(carla, world, blueprints, vehicle, cameras, args, actors)
        _lidar, lidar_q = _spawn_lidar(carla, world, blueprints, vehicle, args, actors)
        _imu, imu_q = _spawn_imu(carla, world, blueprints, vehicle, args, actors)

        video_camera = None
        video_camera_q = None
        video_size = args.video_image_size or args.image_size
        if args.video_output and args.video_image_size:
            video_bp = _camera_blueprint(blueprints, video_size, args.camera_fov, args.hz)
            video_camera = world.spawn_actor(video_bp, _camera_transform(carla, "front"), attach_to=vehicle)
            actors.append(video_camera)
            video_camera_q = queue.Queue()
            video_camera.listen(video_camera_q.put)
        video_writer = _open_video_writer(args.video_output, video_size, args.hz, args.video_scale, args.video_codec)

        infractions = _new_infraction_log()
        collision_bp = blueprints.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)

        def on_collision(event):
            other_type = event.other_actor.type_id if event.other_actor else "unknown"
            loc = event.transform.location
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z)
            infractions[_collision_key(other_type)].append(f"Collision with {other_type} at {_location_text(loc)} intensity={intensity:.2f}")

        collision_sensor.listen(on_collision)

        for _ in range(max(int(args.warmup_sec * args.hz), 0)):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()

        max_steps = int(args.duration_sec * args.hz)
        cross_track_errors = []
        progress_values = []
        success = False
        route_deviation = False
        previous_route_idx = args.start_index
        control_state = {}
        for step in range(max_steps):
            frame = world.tick()
            image_by_name = {
                name: _carla_image_to_rgb(_get_matching(sensor_q, frame), args.image_size)
                for name, sensor_q in camera_queues.items()
            }
            images = np.stack([image_by_name[name] for name in cameras], axis=0)
            lidar_bev = _carla_lidar_to_bev(_get_matching(lidar_q, frame), args)
            imu_data = _get_matching(imu_q, frame)
            video_image = (
                _carla_image_to_rgb(_get_matching(video_camera_q, frame), video_size)
                if video_camera_q is not None
                else image_by_name[cameras[0]]
            )

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
            scalar, nearest_idx, route_dist = _make_scalar_monotonic(route, route_len, odom, imu_values, True, True, previous_route_idx, args)
            previous_route_idx = nearest_idx
            traj, speeds, stop_prob = _predict_token(model, norm, device, scalar, images, lidar_bev)
            _apply_control(carla, vehicle, traj, speeds, stop_prob, args, control_state)

            cross_track_errors.append(route_dist)
            progress_m = float(route[nearest_idx, 3])
            progress_values.append(progress_m)
            route_completion_pct = 100.0 * min(max(progress_m / max(route_len, 1e-6), 0.0), 1.0)
            scores_now = _score_like_leaderboard(route_completion_pct, infractions)
            if video_writer is not None:
                frame_bgr = _render_video_frame(video_image, route, odom, traj, progress_m, route_len, route_dist, scores_now, step, args)
                cv2.putText(frame_bgr, f"stop {stop_prob:.2f}  v_pred {speeds[min(args.control_point_index, len(speeds)-1)]:.2f}", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"stop {stop_prob:.2f}  v_pred {speeds[min(args.control_point_index, len(speeds)-1)]:.2f}", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                video_writer.write(frame_bgr)

            if step == 0 or (step + 1) % max(int(args.report_every_sec * args.hz), 1) == 0:
                print(
                    f"step={step + 1}/{max_steps} route={route_completion_pct:.1f}% "
                    f"cte={route_dist:.2f}m speed={odom[3]:.2f} "
                    f"cmd_v={control_state.get('desired_speed', 0.0):.2f} "
                    f"stop={stop_prob:.2f}",
                    flush=True,
                )
            if route_len - route[nearest_idx, 3] <= args.goal_tolerance_m:
                success = True
                break
            if route_dist > args.failure_distance_m:
                route_deviation = True
                infractions["route_dev"].append(f"Agent deviated from route at {_location_text(location)} distance={route_dist:.2f}m")
                break

        if not success and not route_deviation and len(cross_track_errors) >= max_steps:
            infractions["route_timeout"].append(f"Route timeout after {args.duration_sec:.1f}s")
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        final_progress = progress_values[-1] if progress_values else 0.0
        max_progress = max(progress_values) if progress_values else 0.0
        route_completion_pct = 100.0 * min(max(max_progress / max(route_len, 1e-6), 0.0), 1.0)
        scores = _score_like_leaderboard(route_completion_pct, infractions)
        metrics = {
            "route_source": str(episode_dir),
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "status": "Completed" if success else ("Failed - Agent deviated from the route" if route_deviation else "Failed - Route timeout"),
            "success": success,
            "steps": len(cross_track_errors),
            "infractions": infractions,
            "scores": scores,
            "route_length_m": route_len,
            "final_progress_m": final_progress,
            "max_progress_m": max_progress,
            "route_completion_pct": route_completion_pct,
            "mean_cross_track_error_m": float(np.mean(cross_track_errors)) if cross_track_errors else None,
            "max_cross_track_error_m": float(np.max(cross_track_errors)) if cross_track_errors else None,
            "device": str(device),
            "video_output": args.video_output or None,
        }
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    finally:
        if video_writer is not None:
            video_writer.release()
        _destroy_actors(client, carla, actors)
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Closed-loop CARLA rollout for a Teach2Drive token-fusion model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--route-source", required=True, help="Token episode directory or token_index.npz.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=30)
    parser.add_argument("--spawn-z", type=float, default=0.6)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--camera-fov", type=float, default=90.0)
    parser.add_argument("--bev-size", type=int, default=128)
    parser.add_argument("--x-min", type=float, default=-8.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-min", type=float, default=-14.0)
    parser.add_argument("--y-max", type=float, default=14.0)
    parser.add_argument("--z-min", type=float, default=-2.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--lidar-channels", type=int, default=32)
    parser.add_argument("--lidar-range", type=float, default=60.0)
    parser.add_argument("--lidar-points-per-second", type=int, default=180000)
    parser.add_argument("--lidar-upper-fov", type=float, default=10.0)
    parser.add_argument("--lidar-lower-fov", type=float, default=-25.0)
    parser.add_argument("--lookahead-m", type=float, default=8.0)
    parser.add_argument("--heading-score-weight", type=float, default=0.2)
    parser.add_argument("--control-point-index", type=int, default=1)
    parser.add_argument("--control-horizon-sec", type=float, default=1.0)
    parser.add_argument("--wheelbase-m", type=float, default=2.8)
    parser.add_argument("--max-steer-rad", type=float, default=0.6)
    parser.add_argument("--min-speed", type=float, default=0.4)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--speed-head-mix", type=float, default=0.0)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--brake-kp", type=float, default=0.5)
    parser.add_argument("--max-throttle", type=float, default=0.45)
    parser.add_argument("--max-brake", type=float, default=0.7)
    parser.add_argument("--max-accel-mps2", type=float, default=1.5)
    parser.add_argument("--max-decel-mps2", type=float, default=2.5)
    parser.add_argument("--steer-smoothing", type=float, default=0.65)
    parser.add_argument("--use-stop-head", action="store_true")
    parser.add_argument("--stop-prob-threshold", type=float, default=0.95)
    parser.add_argument("--goal-tolerance-m", type=float, default=5.0)
    parser.add_argument("--failure-distance-m", type=float, default=12.0)
    parser.add_argument("--route-search-back", type=int, default=15)
    parser.add_argument("--route-search-ahead", type=int, default=300)
    parser.add_argument("--report-every-sec", type=float, default=5.0)
    parser.add_argument("--video-output", default="")
    parser.add_argument("--video-image-size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-scale", type=float, default=1.0)
    parser.add_argument("--video-codec", default="mp4v")
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    rollout(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
