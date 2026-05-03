import argparse
import json
import math
import queue
import random
import time
from pathlib import Path

import cv2
import numpy as np


def _import_carla():
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(
            "CARLA Python API is not installed. Install the client version that matches your CARLA server, "
            "for example: pip install carla==0.9.15"
        ) from exc
    return carla


def _carla_image_to_rgb(image, image_size):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    if (image.width, image.height) != tuple(image_size):
        rgb = cv2.resize(rgb, tuple(image_size), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)


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


def _carla_lidar_to_bev(lidar_data, args):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape((-1, 4))
    return _points_to_bev(points, args.bev_size, args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)


def _get_matching(sensor_queue, frame, timeout=2.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise queue.Empty
        data = sensor_queue.get(timeout=remaining)
        if data.frame >= frame:
            return data


def _projected_speed(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    yaw = math.radians(transform.rotation.yaw)
    return float(velocity.x * math.cos(yaw) + velocity.y * math.sin(yaw))


def _safe_stop_actor(actor):
    try:
        if hasattr(actor, "stop"):
            actor.stop()
    except RuntimeError:
        pass


def _destroy_actors(client, carla, actors):
    for actor in reversed(actors):
        _safe_stop_actor(actor)

    destroy_commands = []
    for actor in reversed(actors):
        try:
            destroy_commands.append(carla.command.DestroyActor(actor.id))
        except RuntimeError:
            pass
    if destroy_commands:
        try:
            client.apply_batch_sync(destroy_commands, True)
        except RuntimeError:
            pass


def collect(args):
    carla = _import_carla()
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    if args.map:
        world = client.load_world(args.map)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager(args.traffic_manager_port)
    actors = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.hz
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_random_device_seed(args.seed)

        blueprints = world.get_blueprint_library()
        vehicle_bp = blueprints.filter(args.vehicle_filter)[0]
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "teach2drive_ego")
        spawn_points = world.get_map().get_spawn_points()
        rng = random.Random(args.seed)
        spawn = spawn_points[args.spawn_index] if args.spawn_index >= 0 else rng.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.ignore_lights_percentage(vehicle, args.ignore_lights_percent)

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

        warmup = int(args.warmup_sec * args.hz)
        total = int(args.duration_sec * args.hz)
        for _ in range(warmup):
            world.tick()

        times = []
        poses = []
        images = []
        lidar_bevs = []
        imus = []
        start_elapsed = None

        for i in range(total):
            frame = world.tick()
            snapshot = world.get_snapshot()
            sim_time = float(snapshot.timestamp.elapsed_seconds)
            if start_elapsed is None:
                start_elapsed = sim_time
            camera_data = _get_matching(camera_q, frame)
            lidar_data = _get_matching(lidar_q, frame)
            imu_data = _get_matching(imu_q, frame)

            transform = vehicle.get_transform()
            location = transform.location
            yaw = math.radians(transform.rotation.yaw)
            angular_velocity = vehicle.get_angular_velocity()
            # CARLA reports angular velocity in degrees/sec.
            yaw_rate = math.radians(float(angular_velocity.z))

            times.append(sim_time - start_elapsed)
            poses.append([location.x, location.y, yaw, _projected_speed(vehicle), yaw_rate])
            images.append(_carla_image_to_rgb(camera_data, args.image_size))
            lidar_bevs.append(_carla_lidar_to_bev(lidar_data, args))
            imus.append([
                imu_data.accelerometer.x,
                imu_data.accelerometer.y,
                imu_data.accelerometer.z,
                imu_data.gyroscope.x,
                imu_data.gyroscope.y,
                imu_data.gyroscope.z,
            ])

            if (i + 1) % max(args.hz * 5, 1) == 0:
                print(f"collected {i + 1}/{total} frames")

        poses_np = np.asarray(poses, dtype=np.float32)
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "source": "carla",
            "host": args.host,
            "port": args.port,
            "map": world.get_map().name,
            "hz": args.hz,
            "spawn_index": args.spawn_index,
            "vehicle_filter": args.vehicle_filter,
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
        }
        np.savez_compressed(
            out_path,
            time=np.asarray(times, dtype=np.float32),
            x=poses_np[:, 0],
            y=poses_np[:, 1],
            yaw=poses_np[:, 2],
            v=poses_np[:, 3],
            w=poses_np[:, 4],
            imu=np.asarray(imus, dtype=np.float32),
            imu_valid=np.ones(len(times), dtype=bool),
            images=np.stack(images).astype(np.uint8),
            image_valid=np.ones(len(times), dtype=bool),
            lidar_bev=np.stack(lidar_bevs).astype(np.float16),
            lidar_valid=np.ones(len(times), dtype=bool),
            meta=json.dumps(meta, ensure_ascii=False),
        )
        print(json.dumps({"output": str(out_path), "frames": len(times), "duration_sec": args.duration_sec, "map": world.get_map().name}, indent=2))
    finally:
        _destroy_actors(client, carla, actors)
        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect a Teach2Drive sensor_route.npz from CARLA autopilot.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--map", default="", help="Optional CARLA map name, e.g. Town03.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--warmup-sec", type=float, default=3.0)
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-index", type=int, default=-1)
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--ignore-lights-percent", type=float, default=0.0)
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
    parser.add_argument("--no-rendering", action="store_true")
    return parser


def main():
    collect(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
