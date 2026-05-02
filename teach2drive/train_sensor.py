import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import SensorFusionPolicy


class SensorRouteDataset(Dataset):
    def __init__(self, arrays, indices, scalar_mean, scalar_std, target_mean, target_std, image_dropout=0.0, lidar_dropout=0.0):
        self.scalar = arrays["scalar_features"].astype(np.float32)
        self.targets = arrays["targets"].astype(np.float32)
        self.base_indices = arrays["base_indices"].astype(np.int64)
        self.images = arrays["images"]
        self.lidar = arrays["lidar_bev"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scalar_mean = scalar_mean.astype(np.float32)
        self.scalar_std = scalar_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = target_std.astype(np.float32)
        self.image_dropout = image_dropout
        self.lidar_dropout = lidar_dropout

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        base = int(self.base_indices[idx])
        scalar = (self.scalar[idx] - self.scalar_mean) / self.scalar_std
        target = (self.targets[idx] - self.target_mean) / self.target_std
        image = self.images[base].astype(np.float32).transpose(2, 0, 1) / 255.0
        lidar = self.lidar[base].astype(np.float32)
        if self.image_dropout > 0 and np.random.random() < self.image_dropout:
            image *= 0.0
            scalar[8] = 0.0
        if self.lidar_dropout > 0 and np.random.random() < self.lidar_dropout:
            lidar *= 0.0
            scalar[9] = 0.0
        return torch.from_numpy(scalar), torch.from_numpy(image), torch.from_numpy(lidar), torch.from_numpy(target)


def _safe_std(values):
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return std


def _weighted_mse(pred, target):
    weights = torch.ones_like(target)
    weights[:, 2::3] = 0.35
    return torch.mean(weights * (pred - target) ** 2)


def train(args):
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(Path(args.data).expanduser(), allow_pickle=False)
    scalar = data["scalar_features"].astype(np.float32)
    targets = data["targets"].astype(np.float32)
    base_indices = data["base_indices"].astype(np.int64)
    meta = json.loads(data["meta"].item()) if "meta" in data.files else {}

    rng = np.random.default_rng(args.seed)
    if args.split_mode == "group_random":
        groups = rng.permutation(np.unique(base_indices))
        val_group_count = max(1, int(len(groups) * args.val_ratio))
        val_groups = set(groups[:val_group_count].tolist())
        val_mask = np.asarray([idx in val_groups for idx in base_indices], dtype=bool)
        val_idx = np.nonzero(val_mask)[0]
        train_idx = np.nonzero(~val_mask)[0]
    else:
        order = rng.permutation(len(scalar))
        val_count = max(1, int(len(order) * args.val_ratio))
        val_idx = order[:val_count]
        train_idx = order[val_count:]

    scalar_mean = scalar[train_idx].mean(axis=0)
    scalar_std = _safe_std(scalar[train_idx])
    target_mean = targets[train_idx].mean(axis=0)
    target_std = _safe_std(targets[train_idx])

    train_ds = SensorRouteDataset(data, train_idx, scalar_mean, scalar_std, target_mean, target_std, args.image_dropout, args.lidar_dropout)
    val_ds = SensorRouteDataset(data, val_idx, scalar_mean, scalar_std, target_mean, target_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SensorFusionPolicy(
        scalar_dim=scalar.shape[1],
        output_dim=targets.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for scalar_b, image_b, lidar_b, target_b in train_loader:
            scalar_b = scalar_b.to(device)
            image_b = image_b.to(device)
            lidar_b = lidar_b.to(device)
            target_b = target_b.to(device)
            pred = model(scalar_b, image_b, lidar_b)
            loss = _weighted_mse(pred, target_b)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * len(scalar_b)
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scalar_b, image_b, lidar_b, target_b in val_loader:
                scalar_b = scalar_b.to(device)
                image_b = image_b.to(device)
                lidar_b = lidar_b.to(device)
                target_b = target_b.to(device)
                val_loss += float(_weighted_mse(model(scalar_b, image_b, lidar_b), target_b).item()) * len(scalar_b)
        val_loss /= len(val_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "scalar_mean": scalar_mean,
                "scalar_std": scalar_std,
                "target_mean": target_mean,
                "target_std": target_std,
                "meta": meta,
                "epoch": epoch,
                "val_loss": val_loss,
                "model": {
                    "scalar_dim": int(scalar.shape[1]),
                    "output_dim": int(targets.shape[1]),
                    "embed_dim": args.embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
            }, out_dir / "best_sensor_model.pt")

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

    checkpoint = torch.load(out_dir / "best_sensor_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    preds = []
    target_vals = []
    with torch.no_grad():
        for scalar_b, image_b, lidar_b, target_b in val_loader:
            pred = model(scalar_b.to(device), image_b.to(device), lidar_b.to(device)).cpu().numpy()
            preds.append(pred)
            target_vals.append(target_b.numpy())
    pred_norm = np.concatenate(preds, axis=0)
    target_norm = np.concatenate(target_vals, axis=0)
    pred = pred_norm * target_std + target_mean
    target = target_norm * target_std + target_mean
    pred_r = pred.reshape(len(pred), -1, 3)
    target_r = target.reshape(len(target), -1, 3)
    xy_error = np.linalg.norm(pred_r[:, :, :2] - target_r[:, :, :2], axis=2)
    yaw_error = np.abs((pred_r[:, :, 2] - target_r[:, :, 2] + np.pi) % (2 * np.pi) - np.pi)

    metrics = {
        "data": str(Path(args.data).expanduser()),
        "device": str(device),
        "samples": int(len(scalar)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "split_mode": args.split_mode,
        "mean_xy_error_m_by_horizon": xy_error.mean(axis=0).tolist(),
        "mean_yaw_error_rad_by_horizon": yaw_error.mean(axis=0).tolist(),
        "final_horizon_xy_error_m": float(xy_error[:, -1].mean()),
    }
    (out_dir / "sensor_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "sensor_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "sensor_val_predictions.npz", pred=pred.astype(np.float32), target=target.astype(np.float32), val_idx=val_idx)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train camera/LiDAR route-conditioned policy.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--image-dropout", type=float, default=0.1)
    parser.add_argument("--lidar-dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["group_random", "random"], default="group_random")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

