# Local Artifacts

Training artifacts are intentionally not committed to git.

Current local outputs:

- `runs/indoor_3rd/best_model.pt`
- `runs/indoor_3rd/metrics.json`
- `runs/camera_overlap/best_sensor_model.pt`
- `runs/camera_overlap/sensor_metrics.json`

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

