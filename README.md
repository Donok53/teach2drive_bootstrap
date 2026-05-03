# Teach2Drive Bootstrap

Mapless, demonstration-derived route-following baseline for low-resource end-to-end driving experiments.

This first implementation follows the research policy we settled on:

- no HD map dependency
- no platform-specific action label dependency
- `/cmd_vel` is not used as the training target
- a recorded odometry trajectory becomes the route memory
- each sample is conditioned on a local lookahead goal
- the model predicts future ego-motion from odometry-derived supervision
- pose perturbation augmentation teaches near-route rejoin behavior

## Pipeline

```bash
# 1. Extract odometry and optional IMU from a ROS1 bag.
/usr/bin/python3 -m teach2drive.rosbag_extract \
  --bag "/home/byeongjae/bagfiles/3차 실내 주행/3차 실내주행.bag" \
  --output runs/indoor_3rd/extracted_route.npz

# 2. Build supervised route-following samples.
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.route_dataset \
  --input runs/indoor_3rd/extracted_route.npz \
  --output runs/indoor_3rd/train_dataset.npz \
  --lookahead-m 2.0 \
  --augmentations 4

# 3. Train the bootstrap policy.
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train \
  --data runs/indoor_3rd/train_dataset.npz \
  --out-dir runs/indoor_3rd \
  --epochs 120
```

## Camera/LiDAR Stage

For bags with camera or LiDAR topics, use the sensor-conditioned pipeline:

```bash
/usr/bin/python3 -m teach2drive.sensor_extract \
  --bag "/home/byeongjae/bagfiles/3차 실내 주행/camera_약간겹침.bag" \
  --output runs/camera_overlap/sensor_route.npz

/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.sensor_dataset \
  --input runs/camera_overlap/sensor_route.npz \
  --output runs/camera_overlap/sensor_dataset.npz \
  --lookahead-m 2.0 \
  --augmentations 2 \
  --require-exteroceptive

/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train_sensor \
  --data runs/camera_overlap/sensor_dataset.npz \
  --out-dir runs/camera_overlap \
  --epochs 60
```

## Live ROS1 Inference

Start in dry-run mode first. This publishes only the predicted local trajectory:

```bash
cd /home/byeongjae/code/teach2drive_bootstrap
source /opt/ros/noetic/setup.bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.live_ros_node \
  --checkpoint runs/camera_overlap/best_sensor_model.pt \
  --route-npz runs/camera_overlap/sensor_route.npz
```

Inspect:

```bash
rostopic echo /teach2drive/predicted_path
```

Only after the predicted path looks sane in RViz, enable `/cmd_vel` output with conservative limits:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.live_ros_node \
  --checkpoint runs/camera_overlap/best_sensor_model.pt \
  --route-npz runs/camera_overlap/sensor_route.npz \
  --publish-cmd-vel \
  --max-speed 0.15 \
  --max-yaw-rate 0.4
```

Keep an external emergency stop active. The node publishes zero velocity if odom is stale, camera/LiDAR is stale, or the robot is farther than `--max-route-distance` from the demonstrated route.

## CARLA Check

This repository can also collect a CARLA route dataset and run an initial closed-loop route-following check.

Prerequisites:

```bash
# Use the CARLA client version that matches your server.
/home/byeongjae/miniconda3/envs/vad/bin/pip install carla==0.9.15

# In another terminal, start CARLA, for example:
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

Collect an autopilot demonstration:

```bash
cd /home/byeongjae/code/teach2drive_bootstrap
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_collect \
  --map Town03 \
  --output runs/carla_town03/sensor_route.npz \
  --duration-sec 120 \
  --hz 10
```

If CARLA is already running on a loaded map and `load_world()` is unstable, omit `--map`. The committed smoke-check artifacts in `runs/carla_default` were collected this way from `Town10HD_Opt`.

Build samples and train:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.sensor_dataset \
  --input runs/carla_town03/sensor_route.npz \
  --output runs/carla_town03/sensor_dataset.npz \
  --lookahead-m 6.0 \
  --augmentations 2 \
  --require-exteroceptive

/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train_sensor \
  --data runs/carla_town03/sensor_dataset.npz \
  --out-dir runs/carla_town03 \
  --epochs 80
```

Run a closed-loop rollout from the demonstrated route start:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_rollout \
  --checkpoint runs/carla_town03/best_sensor_model.pt \
  --route-npz runs/carla_town03/sensor_route.npz \
  --output runs/carla_town03/rollout_metrics.json \
  --duration-sec 90 \
  --lookahead-m 6.0
```

To save a front-camera video and lightweight CARLA-Leaderboard-style metrics, add `--video-output`:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_rollout \
  --checkpoint runs/carla_default/best_sensor_model.pt \
  --route-npz runs/carla_default/sensor_route.npz \
  --output runs/carla_default/leaderboard_like_metrics.json \
  --video-output runs/carla_default/rollout_video.mp4 \
  --duration-sec 90 \
  --lookahead-m 6.0 \
  --count-red-lights
```

This evaluator reports route completion, infraction penalty, driving score, collisions, lane invasions, route deviation, timeout, and an annotated video. It is modeled after CARLA Leaderboard scoring, but it is not an official Leaderboard submission run because it does not use ScenarioRunner route XML/scenario configs.

The default visual rollout uses the same low-resolution camera as the model input. For a clearer video while keeping model inference at the trained input size, attach a separate high-resolution video camera:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_rollout \
  --checkpoint runs/carla_default/best_sensor_model.pt \
  --route-npz runs/carla_default/sensor_route.npz \
  --output runs/carla_default/leaderboard_like_metrics_hd.json \
  --video-output runs/carla_default/rollout_video_hd.mp4 \
  --video-image-size 960 540 \
  --video-scale 1.0 \
  --duration-sec 90 \
  --lookahead-m 6.0 \
  --count-red-lights
```

## Model Input

The current baseline uses a compact route-policy input:

- ego velocity and yaw rate from odometry
- optional IMU summary resampled to odometry timestamps
- optional camera image embedding
- optional LiDAR/LaserScan BEV embedding
- local lookahead waypoint in the ego frame
- nearest route anchor in the ego frame
- progress and remaining route distance

This keeps the first version intentionally small while still allowing camera and LiDAR-conditioned policies when those sensors are available.

## Model Output

The model predicts future ego trajectory deltas:

```text
[dx, dy, dyaw] at 0.5s, 1.0s, 1.5s, 2.0s
```

Those outputs are motion-policy targets, not raw actuator commands. A platform-specific controller can later convert the predicted local trajectory into `/cmd_vel`, Ackermann control, or another command interface.
