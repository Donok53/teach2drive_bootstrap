# Local Artifacts

Training artifacts are committed because the live robot node needs the trained checkpoint and route memory.

Current outputs:

- `runs/indoor_3rd/best_model.pt`
- `runs/indoor_3rd/metrics.json`
- `runs/camera_overlap/best_sensor_model.pt`
- `runs/camera_overlap/sensor_metrics.json`
- `runs/carla_default/best_sensor_model.pt`
- `runs/carla_default/sensor_metrics.json`
- `runs/carla_default/rollout_metrics_90s.json`
- `runs/carla_default/leaderboard_like_metrics.json`
- `runs/carla_default/rollout_video.mp4`
- `runs/carla_default/rollout_contact_sheet.jpg`

The sensor model was trained with:

```text
bag: /home/byeongjae/bagfiles/3차 실내 주행/camera_약간겹침.bag
topics: /camera/color/image_raw, /ouster/points, /imu/data, /odom
device: cuda
samples: 5,295
best_val_loss: 0.0013638843024958564
final_horizon_xy_error_m: 0.020436229184269905
```

Recreate the artifacts with the commands in `README.md`.

CARLA smoke-check output:

```text
map: Town10HD_Opt
frames: 300 at 10 Hz
samples: 840
device: cuda
best_val_loss: 0.00962936133146286
90s rollout success: true
90s mean_cross_track_error_m: 0.2168555871537299
90s max_cross_track_error_m: 1.4241096601971066
```

CARLA leaderboard-like visual rollout:

```text
video: runs/carla_default/rollout_video.mp4
preview: runs/carla_default/rollout_contact_sheet.jpg
metrics: runs/carla_default/leaderboard_like_metrics.json
status: Failed - Route timeout
route_completion_pct: 92.535533945104
score_penalty: 0.7
score_composed: 64.7748737615728
mean_cross_track_error_m: 0.2073218737856596
max_cross_track_error_m: 1.4147435682324716
infractions: 1 red_light approximation, 1 route_timeout, 0 collisions, 0 lane invasions
```
