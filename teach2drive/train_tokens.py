import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import TokenFusionPolicy


class TokenRouteDataset(Dataset):
    def __init__(
        self,
        arrays,
        indices,
        scalar_mean,
        scalar_std,
        target_mean,
        target_std,
        control_mean,
        control_std,
        lane_mean,
        lane_std,
        camera_dropout=0.0,
        lidar_dropout=0.0,
    ):
        self.scalar = arrays["scalar_features"].astype(np.float32)
        self.traj_targets = arrays["traj_targets"].astype(np.float32)
        self.speed_targets = arrays["speed_targets"].astype(np.float32)
        self.stop_targets = arrays["stop_targets"].astype(np.float32)
        sample_count = len(self.scalar)
        self.stop_state_targets = arrays["stop_state_targets"].astype(np.int64) if "stop_state_targets" in arrays.files else np.zeros(sample_count, dtype=np.int64)
        self.stop_reason_targets = arrays["stop_reason_targets"].astype(np.int64) if "stop_reason_targets" in arrays.files else np.zeros(sample_count, dtype=np.int64)
        self.stop_reason_masks = arrays["stop_reason_masks"].astype(np.float32) if "stop_reason_masks" in arrays.files else np.zeros(sample_count, dtype=np.float32)
        self.control_targets = arrays["control_targets"].astype(np.float32) if "control_targets" in arrays.files else np.zeros((sample_count, 3), dtype=np.float32)
        self.control_masks = arrays["control_masks"].astype(np.float32) if "control_masks" in arrays.files else np.zeros(sample_count, dtype=np.float32)
        self.lane_targets = arrays["lane_targets"].astype(np.float32) if "lane_targets" in arrays.files else np.zeros((sample_count, 2), dtype=np.float32)
        self.lane_masks = arrays["lane_masks"].astype(np.float32) if "lane_masks" in arrays.files else np.zeros(sample_count, dtype=np.float32)
        self.sample_weights = arrays["sample_weights"].astype(np.float32) if "sample_weights" in arrays.files else np.ones(sample_count, dtype=np.float32)
        self.sample_episode_indices = arrays["sample_episode_indices"].astype(np.int64)
        self.sample_frame_indices = arrays["sample_frame_indices"].astype(np.int64)
        self.episode_dirs = [Path(str(item)) for item in arrays["episode_dirs"]]
        self.cameras = [str(item) for item in arrays["cameras"]]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scalar_mean = scalar_mean.astype(np.float32)
        self.scalar_std = scalar_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        self.control_mean = control_mean.astype(np.float32)
        self.control_std = control_std.astype(np.float32)
        self.lane_mean = lane_mean.astype(np.float32)
        self.lane_std = lane_std.astype(np.float32)
        self.camera_dropout = camera_dropout
        self.lidar_dropout = lidar_dropout
        self.episode_records = [self._read_episode_records(path / "frames.jsonl") for path in self.episode_dirs]

    @staticmethod
    def _read_episode_records(path: Path) -> List[Dict]:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda item: (int(item.get("step", 0)), float(item.get("time", 0.0))))
        return records

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image token: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32).transpose(2, 0, 1) / 255.0

    @staticmethod
    def _read_lidar(path: Path) -> np.ndarray:
        lidar = np.load(path).astype(np.float32)
        if lidar.ndim == 2:
            lidar = lidar[None, :, :]
        elif lidar.ndim == 3 and lidar.shape[-1] in (1, 3) and lidar.shape[0] not in (1, 3):
            lidar = lidar.transpose(2, 0, 1)
        return lidar

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        episode_idx = int(self.sample_episode_indices[idx])
        frame_idx = int(self.sample_frame_indices[idx])
        episode_dir = self.episode_dirs[episode_idx]
        record = self.episode_records[episode_idx][frame_idx]

        scalar = (self.scalar[idx] - self.scalar_mean) / self.scalar_std
        target = np.concatenate([self.traj_targets[idx], self.speed_targets[idx]]).astype(np.float32)
        target = (target - self.target_mean) / self.target_std
        stop = np.asarray([self.stop_targets[idx]], dtype=np.float32)
        stop_state = np.asarray(self.stop_state_targets[idx], dtype=np.int64)
        stop_reason = np.asarray(self.stop_reason_targets[idx], dtype=np.int64)
        stop_reason_mask = np.asarray([self.stop_reason_masks[idx]], dtype=np.float32)
        control = (self.control_targets[idx] - self.control_mean) / self.control_std
        control_mask = np.asarray([self.control_masks[idx]], dtype=np.float32)
        lane = (self.lane_targets[idx] - self.lane_mean) / self.lane_std
        lane_mask = np.asarray([self.lane_masks[idx]], dtype=np.float32)
        sample_weight = np.asarray([self.sample_weights[idx]], dtype=np.float32)

        camera_arrays = []
        dropped_cameras = 0
        for camera in self.cameras:
            image_path = episode_dir / record["camera_tokens"][camera]
            image = self._read_image(image_path)
            if self.camera_dropout > 0 and np.random.random() < self.camera_dropout:
                image *= 0.0
                dropped_cameras += 1
            camera_arrays.append(image)
        cameras = np.stack(camera_arrays).astype(np.float32)
        if dropped_cameras == len(self.cameras):
            scalar[8] = 0.0

        lidar = self._read_lidar(episode_dir / record["lidar_bev_token"])
        if self.lidar_dropout > 0 and np.random.random() < self.lidar_dropout:
            lidar *= 0.0
            scalar[9] = 0.0

        return (
            torch.from_numpy(scalar),
            torch.from_numpy(cameras),
            torch.from_numpy(lidar.astype(np.float32)),
            torch.from_numpy(target),
            torch.from_numpy(stop),
            torch.from_numpy(stop_state.reshape(())),
            torch.from_numpy(stop_reason.reshape(())),
            torch.from_numpy(stop_reason_mask),
            torch.from_numpy(control.astype(np.float32)),
            torch.from_numpy(control_mask),
            torch.from_numpy(lane.astype(np.float32)),
            torch.from_numpy(lane_mask),
            torch.from_numpy(sample_weight),
        )


def _safe_std(values):
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return std


def _safe_masked_mean_std(values, masks, indices):
    masks = masks.astype(bool)
    selected = values[np.asarray(indices, dtype=np.int64)][masks[np.asarray(indices, dtype=np.int64)]]
    if len(selected) == 0:
        return np.zeros(values.shape[1], dtype=np.float32), np.ones(values.shape[1], dtype=np.float32)
    return selected.mean(axis=0).astype(np.float32), _safe_std(selected).astype(np.float32)


def _weighted_average(loss_per_sample, sample_weight):
    weights = sample_weight.reshape(-1)
    return torch.sum(loss_per_sample * weights) / torch.clamp(torch.sum(weights), min=1e-6)


def _trajectory_mse_per_sample(pred, target):
    weights = torch.ones_like(target)
    weights[:, 2::3] = 0.35
    return torch.mean(weights * (pred - target) ** 2, dim=1)


def _masked_weighted_mse(pred, target, mask, sample_weight):
    active = mask.reshape(-1) * sample_weight.reshape(-1)
    loss = torch.mean((pred - target) ** 2, dim=1)
    denom = torch.sum(active)
    if float(denom.detach().cpu()) <= 0.0:
        return pred.sum() * 0.0
    return torch.sum(loss * active) / torch.clamp(denom, min=1e-6)


def _weighted_cross_entropy(logits, target, sample_weight):
    loss = nn.functional.cross_entropy(logits, target.long(), reduction="none")
    return _weighted_average(loss, sample_weight)


def _masked_weighted_cross_entropy(logits, target, mask, sample_weight):
    active = mask.reshape(-1) * sample_weight.reshape(-1)
    loss = nn.functional.cross_entropy(logits, target.long(), reduction="none")
    denom = torch.sum(active)
    if float(denom.detach().cpu()) <= 0.0:
        return logits.sum() * 0.0
    return torch.sum(loss * active) / torch.clamp(denom, min=1e-6)


def _split_indices(args, sample_episode_indices, sample_count):
    rng = np.random.default_rng(args.seed)
    if args.split_mode == "group_random":
        groups = rng.permutation(np.unique(sample_episode_indices))
        if len(groups) >= 2:
            val_group_count = max(1, int(len(groups) * args.val_ratio))
            val_group_count = min(val_group_count, len(groups) - 1)
            val_groups = set(groups[:val_group_count].tolist())
            val_mask = np.asarray([idx in val_groups for idx in sample_episode_indices], dtype=bool)
            return np.nonzero(~val_mask)[0], np.nonzero(val_mask)[0], "group_random"
    order = rng.permutation(sample_count)
    val_count = max(1, int(len(order) * args.val_ratio))
    val_count = min(val_count, len(order) - 1)
    return order[val_count:], order[:val_count], "random"


def _make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train(args):
    index_path = Path(args.index).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(index_path, allow_pickle=False)
    scalar = data["scalar_features"].astype(np.float32)
    traj_targets = data["traj_targets"].astype(np.float32)
    speed_targets = data["speed_targets"].astype(np.float32)
    target = np.concatenate([traj_targets, speed_targets], axis=1).astype(np.float32)
    sample_count = len(scalar)
    stop_state_targets = data["stop_state_targets"].astype(np.int64) if "stop_state_targets" in data.files else np.zeros(sample_count, dtype=np.int64)
    stop_reason_targets = data["stop_reason_targets"].astype(np.int64) if "stop_reason_targets" in data.files else np.zeros(sample_count, dtype=np.int64)
    stop_reason_masks = data["stop_reason_masks"].astype(np.float32) if "stop_reason_masks" in data.files else np.zeros(sample_count, dtype=np.float32)
    control_targets = data["control_targets"].astype(np.float32) if "control_targets" in data.files else np.zeros((sample_count, 3), dtype=np.float32)
    control_masks = data["control_masks"].astype(np.float32) if "control_masks" in data.files else np.zeros(sample_count, dtype=np.float32)
    lane_targets = data["lane_targets"].astype(np.float32) if "lane_targets" in data.files else np.zeros((sample_count, 2), dtype=np.float32)
    lane_masks = data["lane_masks"].astype(np.float32) if "lane_masks" in data.files else np.zeros(sample_count, dtype=np.float32)
    sample_episode_indices = data["sample_episode_indices"].astype(np.int64)
    cameras = [str(item) for item in data["cameras"]]
    meta = json.loads(data["meta"].item()) if "meta" in data.files else {}

    if len(scalar) < 2:
        raise RuntimeError("Need at least two samples for train/val split.")

    train_idx, val_idx, split_mode = _split_indices(args, sample_episode_indices, len(scalar))
    scalar_mean = scalar[train_idx].mean(axis=0)
    scalar_std = _safe_std(scalar[train_idx])
    target_mean = target[train_idx].mean(axis=0)
    target_std = _safe_std(target[train_idx])
    control_mean, control_std = _safe_masked_mean_std(control_targets, control_masks, train_idx)
    lane_mean, lane_std = _safe_masked_mean_std(lane_targets, lane_masks, train_idx)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_ds = TokenRouteDataset(
        data,
        train_idx,
        scalar_mean,
        scalar_std,
        target_mean,
        target_std,
        control_mean,
        control_std,
        lane_mean,
        lane_std,
        camera_dropout=args.camera_dropout,
        lidar_dropout=args.lidar_dropout,
    )
    val_ds = TokenRouteDataset(data, val_idx, scalar_mean, scalar_std, target_mean, target_std, control_mean, control_std, lane_mean, lane_std)
    train_loader = _make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = _make_loader(val_ds, args.batch_size, False, args.num_workers, device)

    traj_dim = int(traj_targets.shape[1])
    speed_dim = int(speed_targets.shape[1])
    target_dim = traj_dim + speed_dim
    control_dim = int(control_targets.shape[1])
    lane_dim = int(lane_targets.shape[1])
    stop_offset = target_dim
    control_offset = stop_offset + 1
    lane_offset = control_offset + control_dim
    stop_state_offset = lane_offset + lane_dim
    stop_reason_offset = stop_state_offset + args.stop_state_classes
    output_dim = stop_reason_offset + args.stop_reason_classes
    model = TokenFusionPolicy(
        scalar_dim=scalar.shape[1],
        output_dim=output_dim,
        num_cameras=len(cameras),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    stop_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        epoch_started_at = time.monotonic()
        step_count = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            (
                scalar_b,
                camera_b,
                lidar_b,
                target_b,
                stop_b,
                stop_state_b,
                stop_reason_b,
                stop_reason_mask_b,
                control_b,
                control_mask_b,
                lane_b,
                lane_mask_b,
                sample_weight_b,
            ) = batch
            scalar_b = scalar_b.to(device)
            camera_b = camera_b.to(device)
            lidar_b = lidar_b.to(device)
            target_b = target_b.to(device)
            stop_b = stop_b.to(device)
            stop_state_b = stop_state_b.to(device)
            stop_reason_b = stop_reason_b.to(device)
            stop_reason_mask_b = stop_reason_mask_b.to(device)
            control_b = control_b.to(device)
            control_mask_b = control_mask_b.to(device)
            lane_b = lane_b.to(device)
            lane_mask_b = lane_mask_b.to(device)
            sample_weight_b = sample_weight_b.to(device)
            pred = model(scalar_b, camera_b, lidar_b)
            traj_loss = _weighted_average(_trajectory_mse_per_sample(pred[:, :traj_dim], target_b[:, :traj_dim]), sample_weight_b)
            speed_loss = _weighted_average(torch.mean((pred[:, traj_dim:target_dim] - target_b[:, traj_dim:target_dim]) ** 2, dim=1), sample_weight_b)
            stop_loss = _weighted_average(stop_loss_fn(pred[:, stop_offset:control_offset], stop_b).reshape(-1), sample_weight_b)
            control_loss = _masked_weighted_mse(pred[:, control_offset:lane_offset], control_b, control_mask_b, sample_weight_b)
            lane_loss = _masked_weighted_mse(pred[:, lane_offset:stop_state_offset], lane_b, lane_mask_b, sample_weight_b)
            stop_state_loss = _weighted_cross_entropy(pred[:, stop_state_offset:stop_reason_offset], stop_state_b, sample_weight_b)
            stop_reason_loss = _masked_weighted_cross_entropy(pred[:, stop_reason_offset:output_dim], stop_reason_b, stop_reason_mask_b, sample_weight_b)
            loss = (
                traj_loss
                + args.speed_loss_weight * speed_loss
                + args.stop_loss_weight * stop_loss
                + args.control_loss_weight * control_loss
                + args.lane_loss_weight * lane_loss
                + args.stop_state_loss_weight * stop_state_loss
                + args.stop_reason_loss_weight * stop_reason_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * len(scalar_b)
            if args.step_log_every > 0 and (step == 1 or step % args.step_log_every == 0 or step == step_count):
                elapsed = time.monotonic() - epoch_started_at
                samples_seen = min(step * args.batch_size, len(train_ds))
                samples_per_sec = samples_seen / max(elapsed, 1e-6)
                running_loss = train_loss / max(samples_seen, 1)
                print(
                    f"epoch={epoch:03d} step={step:05d}/{step_count:05d} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"loss={running_loss:.6f} samples/s={samples_per_sec:.1f}",
                    flush=True,
                )
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (
                    scalar_b,
                    camera_b,
                    lidar_b,
                    target_b,
                    stop_b,
                    stop_state_b,
                    stop_reason_b,
                    stop_reason_mask_b,
                    control_b,
                    control_mask_b,
                    lane_b,
                    lane_mask_b,
                    sample_weight_b,
                ) = batch
                scalar_b = scalar_b.to(device)
                camera_b = camera_b.to(device)
                lidar_b = lidar_b.to(device)
                target_b = target_b.to(device)
                stop_b = stop_b.to(device)
                stop_state_b = stop_state_b.to(device)
                stop_reason_b = stop_reason_b.to(device)
                stop_reason_mask_b = stop_reason_mask_b.to(device)
                control_b = control_b.to(device)
                control_mask_b = control_mask_b.to(device)
                lane_b = lane_b.to(device)
                lane_mask_b = lane_mask_b.to(device)
                sample_weight_b = sample_weight_b.to(device)
                pred = model(scalar_b, camera_b, lidar_b)
                traj_loss = _weighted_average(_trajectory_mse_per_sample(pred[:, :traj_dim], target_b[:, :traj_dim]), sample_weight_b)
                speed_loss = _weighted_average(torch.mean((pred[:, traj_dim:target_dim] - target_b[:, traj_dim:target_dim]) ** 2, dim=1), sample_weight_b)
                stop_loss = _weighted_average(stop_loss_fn(pred[:, stop_offset:control_offset], stop_b).reshape(-1), sample_weight_b)
                control_loss = _masked_weighted_mse(pred[:, control_offset:lane_offset], control_b, control_mask_b, sample_weight_b)
                lane_loss = _masked_weighted_mse(pred[:, lane_offset:stop_state_offset], lane_b, lane_mask_b, sample_weight_b)
                stop_state_loss = _weighted_cross_entropy(pred[:, stop_state_offset:stop_reason_offset], stop_state_b, sample_weight_b)
                stop_reason_loss = _masked_weighted_cross_entropy(pred[:, stop_reason_offset:output_dim], stop_reason_b, stop_reason_mask_b, sample_weight_b)
                loss = (
                    traj_loss
                    + args.speed_loss_weight * speed_loss
                    + args.stop_loss_weight * stop_loss
                    + args.control_loss_weight * control_loss
                    + args.lane_loss_weight * lane_loss
                    + args.stop_state_loss_weight * stop_state_loss
                    + args.stop_reason_loss_weight * stop_reason_loss
                )
                val_loss += float(loss.item()) * len(scalar_b)
        val_loss /= len(val_ds)
        current_lr = scheduler.get_last_lr()[0]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": current_lr})
        (out_dir / "token_history_live.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (out_dir / "token_latest.json").write_text(json.dumps({
            "epoch": epoch,
            "epochs": args.epochs,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
        }, indent=2), encoding="utf-8")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_type": "TokenFusionPolicy",
                "model_state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                "scalar_mean": scalar_mean,
                "scalar_std": scalar_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "control_mean": control_mean,
                "control_std": control_std,
                "lane_mean": lane_mean,
                "lane_std": lane_std,
                "traj_dim": traj_dim,
                "speed_dim": speed_dim,
                "control_dim": control_dim,
                "lane_dim": lane_dim,
                "stop_state_classes": int(args.stop_state_classes),
                "stop_reason_classes": int(args.stop_reason_classes),
                "cameras": cameras,
                "meta": meta,
                "epoch": epoch,
                "val_loss": val_loss,
                "model": {
                    "scalar_dim": int(scalar.shape[1]),
                    "output_dim": int(output_dim),
                    "num_cameras": int(len(cameras)),
                    "embed_dim": int(args.embed_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "transformer_layers": int(args.transformer_layers),
                    "num_heads": int(args.num_heads),
                    "dropout": float(args.dropout),
                },
            }, out_dir / "best_token_model.pt")

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} lr={current_lr:.3e} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}", flush=True)

    checkpoint = torch.load(out_dir / "best_token_model.pt", map_location=device)
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(checkpoint["model_state"])
    model.eval()
    pred_norms = []
    target_norms = []
    stops = []
    stop_logits = []
    stop_state_logits = []
    stop_state_targets_val = []
    stop_reason_logits = []
    stop_reason_targets_val = []
    stop_reason_masks_val = []
    control_norms = []
    control_target_norms = []
    control_masks_val = []
    lane_norms = []
    lane_target_norms = []
    lane_masks_val = []
    with torch.no_grad():
        for batch in val_loader:
            (
                scalar_b,
                camera_b,
                lidar_b,
                target_b,
                stop_b,
                stop_state_b,
                stop_reason_b,
                stop_reason_mask_b,
                control_b,
                control_mask_b,
                lane_b,
                lane_mask_b,
                _sample_weight_b,
            ) = batch
            pred = model(scalar_b.to(device), camera_b.to(device), lidar_b.to(device)).cpu()
            pred_norms.append(pred[:, :target_dim].numpy())
            stop_logits.append(pred[:, stop_offset:control_offset].numpy())
            stop_state_logits.append(pred[:, stop_state_offset:stop_reason_offset].numpy())
            stop_state_targets_val.append(stop_state_b.numpy())
            stop_reason_logits.append(pred[:, stop_reason_offset:output_dim].numpy())
            stop_reason_targets_val.append(stop_reason_b.numpy())
            stop_reason_masks_val.append(stop_reason_mask_b.numpy())
            control_norms.append(pred[:, control_offset:lane_offset].numpy())
            lane_norms.append(pred[:, lane_offset:stop_state_offset].numpy())
            target_norms.append(target_b.numpy())
            stops.append(stop_b.numpy())
            control_target_norms.append(control_b.numpy())
            control_masks_val.append(control_mask_b.numpy())
            lane_target_norms.append(lane_b.numpy())
            lane_masks_val.append(lane_mask_b.numpy())
    pred_norm = np.concatenate(pred_norms, axis=0)
    target_norm = np.concatenate(target_norms, axis=0)
    pred = pred_norm * target_std + target_mean
    target_val = target_norm * target_std + target_mean

    pred_traj = pred[:, :traj_dim].reshape(len(pred), -1, 3)
    target_traj = target_val[:, :traj_dim].reshape(len(target_val), -1, 3)
    xy_error = np.linalg.norm(pred_traj[:, :, :2] - target_traj[:, :, :2], axis=2)
    yaw_error = np.abs((pred_traj[:, :, 2] - target_traj[:, :, 2] + np.pi) % (2 * np.pi) - np.pi)
    speed_error = np.abs(pred[:, traj_dim:target_dim] - target_val[:, traj_dim:target_dim])
    stop_prob = 1.0 / (1.0 + np.exp(-np.concatenate(stop_logits, axis=0)))
    stop_target = np.concatenate(stops, axis=0)
    stop_acc = float(np.mean((stop_prob >= 0.5) == (stop_target >= 0.5)))
    stop_state_pred = np.argmax(np.concatenate(stop_state_logits, axis=0), axis=1)
    stop_state_target = np.concatenate(stop_state_targets_val, axis=0)
    stop_state_acc = float(np.mean(stop_state_pred == stop_state_target))
    stop_reason_pred = np.argmax(np.concatenate(stop_reason_logits, axis=0), axis=1)
    stop_reason_target = np.concatenate(stop_reason_targets_val, axis=0)
    stop_reason_mask_val = np.concatenate(stop_reason_masks_val, axis=0).reshape(-1).astype(bool)
    control_pred = np.concatenate(control_norms, axis=0) * control_std + control_mean
    control_target = np.concatenate(control_target_norms, axis=0) * control_std + control_mean
    control_mask_val = np.concatenate(control_masks_val, axis=0).reshape(-1).astype(bool)
    lane_pred = np.concatenate(lane_norms, axis=0) * lane_std + lane_mean
    lane_target = np.concatenate(lane_target_norms, axis=0) * lane_std + lane_mean
    lane_mask_val = np.concatenate(lane_masks_val, axis=0).reshape(-1).astype(bool)

    metrics = {
        "index": str(index_path),
        "device": str(device),
        "data_parallel": bool(args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1),
        "gpu_count": int(torch.cuda.device_count()) if device.type == "cuda" else 0,
        "episodes": int(len(np.unique(sample_episode_indices))),
        "samples": int(len(scalar)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "split_mode": split_mode,
        "cameras": cameras,
        "horizons": meta.get("horizons", []),
        "mean_xy_error_m_by_horizon": xy_error.mean(axis=0).tolist(),
        "mean_yaw_error_rad_by_horizon": yaw_error.mean(axis=0).tolist(),
        "mean_speed_error_mps_by_horizon": speed_error.mean(axis=0).tolist(),
        "final_horizon_xy_error_m": float(xy_error[:, -1].mean()),
        "stop_accuracy": stop_acc,
        "stop_state_accuracy": stop_state_acc,
        "stop_reason_label_ratio": float(np.mean(stop_reason_masks)),
        "control_label_ratio": float(np.mean(control_masks)),
        "lane_label_ratio": float(np.mean(lane_masks)),
    }
    if np.any(stop_reason_mask_val):
        metrics["stop_reason_accuracy"] = float(np.mean(stop_reason_pred[stop_reason_mask_val] == stop_reason_target[stop_reason_mask_val]))
    if np.any(control_mask_val):
        metrics["control_mae_steer_throttle_brake"] = np.mean(np.abs(control_pred[control_mask_val] - control_target[control_mask_val]), axis=0).tolist()
    if np.any(lane_mask_val):
        metrics["lane_mae_offset_heading"] = np.mean(np.abs(lane_pred[lane_mask_val] - lane_target[lane_mask_val]), axis=0).tolist()
    (out_dir / "token_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "token_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        out_dir / "token_val_predictions.npz",
        pred=pred.astype(np.float32),
        target=target_val.astype(np.float32),
        stop_prob=stop_prob.astype(np.float32),
        stop_target=stop_target.astype(np.float32),
        stop_state_pred=stop_state_pred.astype(np.int64),
        stop_state_target=stop_state_target.astype(np.int64),
        stop_reason_pred=stop_reason_pred.astype(np.int64),
        stop_reason_target=stop_reason_target.astype(np.int64),
        stop_reason_mask=stop_reason_mask_val.astype(np.float32),
        control_pred=control_pred.astype(np.float32),
        control_target=control_target.astype(np.float32),
        control_mask=control_mask_val.astype(np.float32),
        lane_pred=lane_pred.astype(np.float32),
        lane_target=lane_target.astype(np.float32),
        lane_mask=lane_mask_val.astype(np.float32),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a token-fusion camera/LiDAR/IMU/odom policy.")
    parser.add_argument("--index", required=True, help="Token index .npz from teach2drive.token_dataset.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--speed-loss-weight", type=float, default=0.35)
    parser.add_argument("--stop-loss-weight", type=float, default=0.15)
    parser.add_argument("--control-loss-weight", type=float, default=0.25)
    parser.add_argument("--lane-loss-weight", type=float, default=0.15)
    parser.add_argument("--stop-state-loss-weight", type=float, default=0.0)
    parser.add_argument("--stop-reason-loss-weight", type=float, default=0.0)
    parser.add_argument("--stop-state-classes", type=int, default=4)
    parser.add_argument("--stop-reason-classes", type=int, default=8)
    parser.add_argument("--camera-dropout", type=float, default=0.05)
    parser.add_argument("--lidar-dropout", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["group_random", "random"], default="group_random")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-parallel", action="store_true", help="Use torch.nn.DataParallel across visible CUDA devices.")
    parser.add_argument("--step-log-every", type=int, default=200, help="Print training progress every N batches. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
