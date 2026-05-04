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

## Tokenized CARLA Collection

For larger multi-episode datasets, use the tokenized collector instead of the single `.npz` collector. It stores one episode per directory with front/left/right camera tokens, LiDAR BEV tokens, IMU, odometry, and per-frame metadata:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_collect_tokens \
  --output-root "data/carla/town10_3cam_640x360" \
  --episodes 24 \
  --drive-sec 300 \
  --stopped-start-sec 3 \
  --stopped-end-sec 8 \
  --hz 10 \
  --image-size 640 360 \
  --cameras front,left,right \
  --bev-size 128 \
  --jpeg-quality 90 \
  --report-every-sec 30
```

Monitor dataset growth in another terminal:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.dataset_rate \
  --path "data/carla/town10_3cam_640x360" \
  --interval-sec 10
```

Build the lightweight training index. This does not copy images into a giant `.npz`; it stores token references plus odom-derived targets, optional label masks, and phase weights:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.token_dataset \
  --input-root data/carla/town10_3cam_640x360 \
  --output runs/town10_3cam_640x360_tokens/token_index.npz \
  --cameras front,left,right \
  --horizons 0.5,1.0,1.5,2.0 \
  --lookahead-m 8.0 \
  --augmentations 2 \
  --lateral-max-m 1.2 \
  --forward-max-m 0.7 \
  --yaw-max-deg 45 \
  --drive-weight 1.0 \
  --stopped-start-weight 0.25 \
  --stopped-end-weight 0.25
```

Train the token-fusion policy:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 /home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train_tokens \
  --index runs/town10_3cam_640x360_tokens/token_index.npz \
  --out-dir runs/town10_3cam_640x360_tokens/train_v1 \
  --epochs 20 \
  --batch-size 16 \
  --lr 5e-4 \
  --embed-dim 160 \
  --hidden-dim 320 \
  --transformer-layers 2 \
  --num-heads 4 \
  --num-workers 4 \
  --step-log-every 200 \
  --log-every 1 2>&1 | tee runs/town10_3cam_640x360_tokens/train_v1/train.log
```

### Rule-Aware Comparison Track

For the stop-go instability experiment, keep the original token imitation run as the baseline and train a second rule-aware model. The rule-aware index adds stop-state labels:

- `drive`
- `approach_stop`
- `stopped_waiting`
- `release_go`

It also supports stop-reason labels when the dataset was collected with the newer token collector metadata:

- `traffic_light`
- `stop_sign`
- `front_vehicle`
- `junction_yield`

Build a separate index so the baseline run remains reproducible:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.token_dataset \
  --input-root data/carla/town10_3cam_640x360 \
  --output runs/town10_3cam_640x360_tokens/token_rule_index.npz \
  --cameras front,left,right \
  --horizons 0.5,1.0,1.5,2.0 \
  --lookahead-m 8.0 \
  --augmentations 2 \
  --lateral-max-m 1.2 \
  --forward-max-m 0.7 \
  --yaw-max-deg 45 \
  --drive-weight 1.0 \
  --stopped-start-weight 0.25 \
  --stopped-end-weight 0.25 \
  --stop-state-stop-speed 0.35 \
  --stop-state-move-speed 1.0
```

Train the rule-aware comparison model:

```bash
mkdir -p runs/town10_3cam_640x360_tokens/train_v3_ruleaware
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 /home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train_ruleaware \
  --index runs/town10_3cam_640x360_tokens/token_rule_index.npz \
  --out-dir runs/town10_3cam_640x360_tokens/train_v3_ruleaware \
  --epochs 20 \
  --batch-size 32 \
  --lr 5e-4 \
  --embed-dim 160 \
  --hidden-dim 320 \
  --transformer-layers 2 \
  --num-heads 4 \
  --num-workers 8 \
  --data-parallel \
  --step-log-every 200 \
  --log-every 1 2>&1 | tee runs/town10_3cam_640x360_tokens/train_v3_ruleaware/train.log
```

### Post-Hoc Pseudo Labels

For the realistic rule-aware setting, keep the raw tokenized dataset unchanged and add `pseudo_labels.jsonl` beside each episode. The pseudo labels are generated offline from raw camera/LiDAR/odom streams:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 /home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.pseudo_label \
  --input-root data/carla/town10_3cam_640x360 \
  --output-name pseudo_labels.jsonl \
  --summary-output runs/town10_3cam_640x360_tokens/pseudo_label_summary.json \
  --camera-teacher yolo \
  --yolo-model yolov8n.pt \
  --yolo-device 0 \
  --yolo-imgsz 640 \
  --yolo-batch 16 \
  --yolo-chunk 128
```

Then build a separate pseudo-rule index:

```bash
/home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.token_dataset \
  --input-root data/carla/town10_3cam_640x360 \
  --output runs/town10_3cam_640x360_tokens/token_pseudo_rule_index.npz \
  --cameras front,left,right \
  --horizons 0.5,1.0,1.5,2.0 \
  --lookahead-m 8.0 \
  --augmentations 2 \
  --lateral-max-m 1.2 \
  --forward-max-m 0.7 \
  --yaw-max-deg 45 \
  --stop-state-stop-speed 0.35 \
  --stop-state-move-speed 1.0 \
  --min-pseudo-reason-confidence 0.25
```

Train with the same rule-aware entry point:

```bash
mkdir -p runs/town10_3cam_640x360_tokens/train_v4_pseudo_ruleaware
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 /home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.train_ruleaware \
  --index runs/town10_3cam_640x360_tokens/token_pseudo_rule_index.npz \
  --out-dir runs/town10_3cam_640x360_tokens/train_v4_pseudo_ruleaware \
  --epochs 20 \
  --batch-size 32 \
  --lr 5e-4 \
  --embed-dim 160 \
  --hidden-dim 320 \
  --transformer-layers 2 \
  --num-heads 4 \
  --num-workers 8 \
  --data-parallel \
  --step-log-every 200 \
  --log-every 1 2>&1 | tee runs/town10_3cam_640x360_tokens/train_v4_pseudo_ruleaware/train.log
```

Run a closed-loop CARLA visualization for a token model:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 /home/byeongjae/miniconda3/envs/vad/bin/python -m teach2drive.carla_rollout_tokens \
  --checkpoint runs/town10_3cam_640x360_tokens/train_v1/best_token_model.pt \
  --route-source data/carla/town10_3cam_640x360/episode_000000 \
  --output runs/town10_3cam_640x360_tokens/train_v1/rollout_metrics.json \
  --video-output runs/town10_3cam_640x360_tokens/train_v1/rollout.mp4 \
  --duration-sec 60 \
  --hz 10 \
  --image-size 640 360 \
  --video-image-size 1280 720
```

## Remote Training With Git

Code is managed with git; datasets and training outputs are intentionally local and ignored by git. On DL3, clone or update the code, then symlink the dataset location into the repository:

```bash
mkdir -p /media/aimlab/HDD00/users/byengjae
cd /media/aimlab/HDD00/users/byengjae

git clone https://github.com/Donok53/teach2drive_bootstrap.git
cd teach2drive_bootstrap

mkdir -p data/carla runs/town10_3cam_640x360_tokens
ln -sfn /media/aimlab/HDD00/datasets/teach2drive/town10_3cam_640x360 \
  data/carla/town10_3cam_640x360
```

After the first clone, update code with:

```bash
cd /media/aimlab/HDD00/users/byengjae/teach2drive_bootstrap
git pull
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

The baseline always predicts future ego trajectory deltas:

```text
[dx, dy, dyaw] at 0.5s, 1.0s, 1.5s, 2.0s
```

The token training path is label-adaptive. Odom-derived trajectory, speed, and stop/move targets are always available. Extra supervision such as CARLA expert control (`steer`, `throttle`, `brake`) or lane-center labels is used only when present; otherwise its mask is zero and that loss is skipped. This keeps camera-only, LiDAR-only, and camera+LiDAR datasets compatible with the same training code.

Those outputs are motion-policy targets, not raw actuator commands. A platform-specific controller can later convert the predicted local trajectory into `/cmd_vel`, Ackermann control, or another command interface.
