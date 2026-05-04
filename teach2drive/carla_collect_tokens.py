import argparse
import json
import math
import queue
import random
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from .carla_collect import _carla_lidar_to_bev, _destroy_actors, _get_matching, _import_carla
from .geometry import wrap_angle


CAMERA_TRANSFORMS = {
    "front": ((1.5, 0.0, 1.6), (0.0, 0.0, 0.0)),
    "left": ((1.0, -0.55, 1.55), (0.0, -45.0, 0.0)),
    "right": ((1.0, 0.55, 1.55), (0.0, 45.0, 0.0)),
}


def _carla_image_to_bgr(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return array[:, :, :3].copy()


def _projected_speed(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    yaw = math.radians(transform.rotation.yaw)
    return float(velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw))


def _camera_blueprint(blueprints, name, image_size, fov, hz):
    bp = blueprints.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(image_size[0]))
    bp.set_attribute("image_size_y", str(image_size[1]))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", str(1.0 / hz))
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", f"teach2drive_{name}")
    return bp


def _camera_transform(carla, name):
    location, rotation = CAMERA_TRANSFORMS[name]
    return carla.Transform(
        carla.Location(x=location[0], y=location[1], z=location[2]),
        carla.Rotation(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]),
    )


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_rate(bytes_written, elapsed_sec):
    mb = bytes_written / (1024 * 1024)
    mbps = mb / max(elapsed_sec, 1e-6)
    gb_per_hour = mbps * 3600 / 1024
    return mb, mbps, gb_per_hour


def _gib_per_sim_hour(bytes_written, frames, hz):
    sim_sec = frames / max(float(hz), 1e-6)
    return bytes_written / (1024 ** 3) * 3600 / max(sim_sec, 1e-6)


def _choose_spawn(spawn_points, episode_idx, args):
    if args.spawn_indices:
        return spawn_points[args.spawn_indices[episode_idx % len(args.spawn_indices)]]
    rng = random.Random(args.seed + episode_idx)
    if args.spawn_index >= 0:
        return spawn_points[(args.spawn_index + episode_idx) % len(spawn_points)]
    return rng.choice(spawn_points)


def _spawn_cameras(carla, world, blueprints, vehicle, args, actors):
    cameras = {}
    queues = {}
    for name in args.cameras:
        bp = _camera_blueprint(blueprints, name, args.image_size, args.camera_fov, args.hz)
        sensor = world.spawn_actor(bp, _camera_transform(carla, name), attach_to=vehicle)
        actors.append(sensor)
        sensor_queue = queue.Queue()
        sensor.listen(sensor_queue.put)
        cameras[name] = sensor
        queues[name] = sensor_queue
    return cameras, queues


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


def _phase_for_step(step, start_steps, drive_steps):
    if step < start_steps:
        return "stopped_start"
    if step < start_steps + drive_steps:
        return "drive"
    return "stopped_end"


def _enum_name(value):
    return str(value).rsplit(".", 1)[-1]


def _lane_features(carla, carla_map, location, yaw):
    try:
        waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    except RuntimeError:
        waypoint = None
    if waypoint is None:
        return {"valid": False}
    wp_location = waypoint.transform.location
    wp_yaw = math.radians(float(waypoint.transform.rotation.yaw))
    dx = float(location.x - wp_location.x)
    dy = float(location.y - wp_location.y)
    lateral_offset = -math.sin(wp_yaw) * dx + math.cos(wp_yaw) * dy
    heading_error = float(wrap_angle(yaw - wp_yaw))
    return {
        "valid": True,
        "road_id": int(waypoint.road_id),
        "section_id": int(waypoint.section_id),
        "lane_id": int(waypoint.lane_id),
        "lane_width": float(waypoint.lane_width),
        "is_junction": bool(waypoint.is_junction),
        "lane_center_offset_m": float(lateral_offset),
        "lane_heading_error_rad": heading_error,
        "lane_type": _enum_name(waypoint.lane_type),
    }


def _stop_sign_features(carla, carla_map, location, args):
    try:
        waypoint = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        landmarks = waypoint.get_landmarks(args.rule_lookahead_m, True) if waypoint is not None else []
    except RuntimeError:
        landmarks = []
    matches = []
    for landmark in landmarks:
        landmark_type = str(getattr(landmark, "type", "")).lower()
        landmark_name = str(getattr(landmark, "name", "")).lower()
        if "stop" not in landmark_type and "stop" not in landmark_name and landmark_type != "206":
            continue
        distance = float(getattr(landmark, "distance", args.rule_lookahead_m))
        matches.append({
            "id": int(getattr(landmark, "id", -1)),
            "type": str(getattr(landmark, "type", "")),
            "name": str(getattr(landmark, "name", "")),
            "distance_m": distance,
        })
    matches.sort(key=lambda item: item["distance_m"])
    return {
        "valid": bool(matches),
        "distance_m": float(matches[0]["distance_m"]) if matches else None,
        "landmarks": matches[:3],
    }


def _front_vehicle_features(carla, world, vehicle, carla_map, location, yaw, args):
    ego_wp = None
    try:
        ego_wp = carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
    except RuntimeError:
        pass
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    left = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    nearest = None
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        other_loc = actor.get_location()
        rel = np.asarray([other_loc.x - location.x, other_loc.y - location.y], dtype=np.float32)
        longitudinal = float(np.dot(rel, forward))
        lateral = abs(float(np.dot(rel, left)))
        if longitudinal <= 0.0 or longitudinal > args.front_vehicle_lookahead_m or lateral > args.front_vehicle_lateral_m:
            continue
        same_lane = None
        if ego_wp is not None:
            try:
                other_wp = carla_map.get_waypoint(other_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                same_lane = bool(
                    other_wp is not None
                    and other_wp.road_id == ego_wp.road_id
                    and other_wp.section_id == ego_wp.section_id
                    and other_wp.lane_id == ego_wp.lane_id
                )
            except RuntimeError:
                same_lane = None
        if same_lane is False:
            continue
        vel = actor.get_velocity()
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y)
        candidate = {
            "id": int(actor.id),
            "type_id": actor.type_id,
            "distance_m": longitudinal,
            "lateral_m": lateral,
            "speed_mps": float(speed),
            "same_lane": same_lane,
        }
        if nearest is None or candidate["distance_m"] < nearest["distance_m"]:
            nearest = candidate
    return {"valid": nearest is not None, **(nearest or {})}


def _record_frame(
    carla,
    world,
    carla_map,
    vehicle,
    camera_queues,
    lidar_q,
    imu_q,
    episode_dir,
    frames_file,
    args,
    episode_token,
    step,
    phase,
    start_elapsed,
):
    frame = world.tick()
    snapshot = world.get_snapshot()
    sim_time = float(snapshot.timestamp.elapsed_seconds)
    if start_elapsed is None:
        start_elapsed = sim_time

    camera_data = {name: _get_matching(sensor_q, frame) for name, sensor_q in camera_queues.items()}
    lidar_data = _get_matching(lidar_q, frame)
    imu_data = _get_matching(imu_q, frame)

    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    yaw = math.radians(rotation.yaw)
    angular_velocity = vehicle.get_angular_velocity()
    yaw_rate = math.radians(float(angular_velocity.z))
    velocity = vehicle.get_velocity()
    control = vehicle.get_control()
    traffic_light = vehicle.get_traffic_light()
    traffic_light_state = vehicle.get_traffic_light_state()

    frame_token = uuid.uuid4().hex
    camera_tokens = {}
    bytes_written = 0
    for name, image in camera_data.items():
        camera_path = episode_dir / "camera" / name / f"{step:06d}_{frame_token}.jpg"
        bgr = _carla_image_to_bgr(image)
        ok = cv2.imwrite(str(camera_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to write image: {camera_path}")
        camera_tokens[name] = str(camera_path.relative_to(episode_dir))
        bytes_written += camera_path.stat().st_size

    lidar_path = episode_dir / "lidar_bev" / f"{step:06d}_{frame_token}.npy"
    lidar_bev = _carla_lidar_to_bev(lidar_data, args)
    np.save(lidar_path, lidar_bev.astype(np.float16))
    bytes_written += lidar_path.stat().st_size

    record = {
        "episode_token": episode_token,
        "frame_token": frame_token,
        "step": int(step),
        "phase": phase,
        "carla_frame": int(frame),
        "time": float(sim_time - start_elapsed),
        "camera_tokens": camera_tokens,
        "lidar_bev_token": str(lidar_path.relative_to(episode_dir)),
        "imu": {
            "accelerometer": [float(imu_data.accelerometer.x), float(imu_data.accelerometer.y), float(imu_data.accelerometer.z)],
            "gyroscope": [float(imu_data.gyroscope.x), float(imu_data.gyroscope.y), float(imu_data.gyroscope.z)],
        },
        "control": {
            "throttle": float(control.throttle),
            "steer": float(control.steer),
            "brake": float(control.brake),
            "hand_brake": bool(control.hand_brake),
            "reverse": bool(control.reverse),
            "manual_gear_shift": bool(control.manual_gear_shift),
            "gear": int(control.gear),
        },
        "traffic_light": {
            "is_at_traffic_light": bool(vehicle.is_at_traffic_light()),
            "id": int(traffic_light.id) if traffic_light is not None else None,
            "state": _enum_name(traffic_light_state),
        },
        "lane": _lane_features(carla, carla_map, location, yaw),
        "stop_sign": _stop_sign_features(carla, carla_map, location, args),
        "front_vehicle": _front_vehicle_features(carla, world, vehicle, carla_map, location, yaw, args),
        "odom": {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
            "roll": math.radians(float(rotation.roll)),
            "pitch": math.radians(float(rotation.pitch)),
            "yaw": float(yaw),
            "v_forward": float(_projected_speed(vehicle)),
            "velocity": [float(velocity.x), float(velocity.y), float(velocity.z)],
            "yaw_rate": float(yaw_rate),
        },
    }
    frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return start_elapsed, bytes_written


def _prepare_episode_dirs(episode_dir, cameras):
    (episode_dir / "lidar_bev").mkdir(parents=True, exist_ok=True)
    for name in cameras:
        (episode_dir / "camera" / name).mkdir(parents=True, exist_ok=True)


def collect_episode(carla, client, world, traffic_manager, episode_idx, args, dataset_meta):
    actors = []
    episode_token = uuid.uuid4().hex
    episode_dir = Path(args.output_root).expanduser() / f"episode_{episode_idx:06d}"
    if episode_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Episode directory exists: {episode_dir}")
    if episode_dir.exists() and args.overwrite:
        shutil.rmtree(episode_dir)
    _prepare_episode_dirs(episode_dir, args.cameras)

    frames_path = episode_dir / "frames.jsonl"
    frames_file = frames_path.open("w", encoding="utf-8", buffering=1)

    try:
        blueprints = world.get_blueprint_library()
        carla_map = world.get_map()
        vehicle_bp = blueprints.filter(args.vehicle_filter)[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "teach2drive_ego")
        spawn = _choose_spawn(world.get_map().get_spawn_points(), episode_idx, args)
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        vehicle.set_autopilot(False, traffic_manager.get_port())
        vehicle.apply_control(carla.VehicleControl(brake=1.0))
        traffic_manager.ignore_lights_percentage(vehicle, args.ignore_lights_percent)

        _cameras, camera_queues = _spawn_cameras(carla, world, blueprints, vehicle, args, actors)
        _lidar, lidar_q = _spawn_lidar(carla, world, blueprints, vehicle, args, actors)
        _imu, imu_q = _spawn_imu(carla, world, blueprints, vehicle, args, actors)

        episode_meta = {
            **dataset_meta,
            "episode_token": episode_token,
            "episode_index": episode_idx,
            "spawn_transform": {
                "location": [spawn.location.x, spawn.location.y, spawn.location.z],
                "rotation": [spawn.rotation.pitch, spawn.rotation.yaw, spawn.rotation.roll],
            },
        }
        _write_json(episode_dir / "episode_meta.json", episode_meta)

        for _ in range(int(args.warmup_sec * args.hz)):
            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            world.tick()

        start_steps = int(args.stopped_start_sec * args.hz)
        drive_steps = int(args.drive_sec * args.hz)
        end_steps = int(args.stopped_end_sec * args.hz)
        total_steps = start_steps + drive_steps + end_steps

        start_elapsed = None
        bytes_written = 0
        started_at = time.monotonic()
        for step in range(total_steps):
            phase = _phase_for_step(step, start_steps, drive_steps)
            if phase == "stopped_start":
                vehicle.apply_control(carla.VehicleControl(brake=1.0))
            elif phase == "drive" and step == start_steps:
                vehicle.set_autopilot(True, traffic_manager.get_port())
            elif phase == "stopped_end" and step == start_steps + drive_steps:
                vehicle.set_autopilot(False, traffic_manager.get_port())
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            elif phase == "stopped_end":
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))

            start_elapsed, frame_bytes = _record_frame(
                carla,
                world,
                carla_map,
                vehicle,
                camera_queues,
                lidar_q,
                imu_q,
                episode_dir,
                frames_file,
                args,
                episode_token,
                step,
                phase,
                start_elapsed,
            )
            bytes_written += frame_bytes
            if (step + 1) % max(int(args.report_every_sec * args.hz), 1) == 0:
                elapsed = time.monotonic() - started_at
                mb, mbps, wall_gbh = _format_rate(bytes_written, elapsed)
                sim_gbh = _gib_per_sim_hour(bytes_written, step + 1, args.hz)
                print(
                    f"episode={episode_idx + 1}/{args.episodes} frames={step + 1}/{total_steps} "
                    f"written={mb:.1f}MiB write_rate={mbps:.2f}MiB/s "
                    f"wall_est={wall_gbh:.1f}GiB/h dataset_est={sim_gbh:.1f}GiB/sim-h"
                )

        elapsed = time.monotonic() - started_at
        _mb, mbps, wall_gbh = _format_rate(bytes_written, elapsed)
        sim_gbh = _gib_per_sim_hour(bytes_written, total_steps, args.hz)
        summary = {
            "episode_index": episode_idx,
            "episode_token": episode_token,
            "frames": total_steps,
            "bytes_written": int(bytes_written),
            "elapsed_wall_sec": elapsed,
            "average_mib_per_sec": mbps,
            "estimated_gib_per_wall_hour": wall_gbh,
            "estimated_gib_per_sim_hour": sim_gbh,
        }
        _write_json(episode_dir / "episode_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
    finally:
        frames_file.close()
        _destroy_actors(client, carla, actors)


def collect(args):
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    carla = _import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if args.map:
        world = client.load_world(args.map)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    summaries = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        dataset_meta = {
            "dataset": "teach2drive_tokenized_carla",
            "map": world.get_map().name,
            "hz": args.hz,
            "cameras": args.cameras,
            "image_size_wh": args.image_size,
            "jpeg_quality": args.jpeg_quality,
            "lidar_bev_size": args.bev_size,
            "drive_sec": args.drive_sec,
            "stopped_start_sec": args.stopped_start_sec,
            "stopped_end_sec": args.stopped_end_sec,
            "policy": "CARLA Traffic Manager autopilot between stopped_start and stopped_end phases",
            "layout": {
                "frames": "episode_xxxxxx/frames.jsonl",
                "camera_tokens": "episode_xxxxxx/camera/{front,left,right}/*.jpg",
                "lidar_tokens": "episode_xxxxxx/lidar_bev/*.npy",
                "meta": "episode_xxxxxx/episode_meta.json",
            },
        }
        _write_json(output_root / "dataset_meta.json", dataset_meta)

        for episode_idx in range(args.episodes):
            summaries.append(collect_episode(carla, client, world, traffic_manager, episode_idx, args, dataset_meta))
            _write_json(output_root / "dataset_summary.json", {"episodes": summaries})
    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def _parse_cameras(value):
    cameras = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in cameras if name not in CAMERA_TRANSFORMS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown cameras: {unknown}. Choose from {sorted(CAMERA_TRANSFORMS)}")
    return cameras


def _parse_spawn_indices(value):
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect multi-episode tokenized CARLA data for Teach2Drive.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--drive-sec", type=float, default=300.0)
    parser.add_argument("--stopped-start-sec", type=float, default=3.0)
    parser.add_argument("--stopped-end-sec", type=float, default=8.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-index", type=int, default=-1)
    parser.add_argument("--spawn-indices", type=_parse_spawn_indices, default=[])
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--ignore-lights-percent", type=float, default=0.0)
    parser.add_argument("--cameras", type=_parse_cameras, default=["front", "left", "right"])
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 360], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--jpeg-quality", type=int, default=90)
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
    parser.add_argument("--report-every-sec", type=float, default=30.0)
    parser.add_argument("--rule-lookahead-m", type=float, default=35.0)
    parser.add_argument("--front-vehicle-lookahead-m", type=float, default=35.0)
    parser.add_argument("--front-vehicle-lateral-m", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rendering", action="store_true")
    return parser


def main():
    collect(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
