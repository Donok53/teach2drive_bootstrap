import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ModelConfig, RoutePolicyMLP


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return std


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.ones_like(target)
    weights[:, 2::3] = 0.35
    return torch.mean(weights * (pred - target) ** 2)


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(data_path, allow_pickle=False)
    x = raw["features"].astype(np.float32)
    y = raw["targets"].astype(np.float32)
    meta = json.loads(raw["meta"].item()) if "meta" in raw.files else {}

    rng = np.random.default_rng(args.seed)
    if "sample_info" in raw.files and args.split_mode == "group_random":
        base_index = raw["sample_info"][:, 0].astype(np.int64)
        groups = rng.permutation(np.unique(base_index))
        val_group_count = max(1, int(len(groups) * args.val_ratio))
        val_groups = set(groups[:val_group_count].tolist())
        val_mask = np.asarray([idx in val_groups for idx in base_index], dtype=bool)
        val_idx = np.nonzero(val_mask)[0]
        train_idx = np.nonzero(~val_mask)[0]
    elif args.split_mode == "time":
        order = np.arange(len(x))
        val_count = max(1, int(len(order) * args.val_ratio))
        train_idx = order[:-val_count]
        val_idx = order[-val_count:]
    else:
        indices = rng.permutation(len(x))
        val_count = max(1, int(len(indices) * args.val_ratio))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]

    x_mean = x[train_idx].mean(axis=0)
    x_std = _safe_std(x[train_idx])
    y_mean = y[train_idx].mean(axis=0)
    y_std = _safe_std(y[train_idx])

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    train_ds = TensorDataset(torch.from_numpy(x_norm[train_idx]), torch.from_numpy(y_norm[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(x_norm[val_idx]), torch.from_numpy(y_norm[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    config = ModelConfig(
        input_dim=x.shape[1],
        output_dim=y.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    )
    model = RoutePolicyMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = _weighted_mse(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
        scheduler.step()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = _weighted_mse(pred, yb)
                val_loss += float(loss.item()) * len(xb)
        val_loss /= len(val_ds)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "meta": meta,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best_model.pt")

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}")

    with torch.no_grad():
        checkpoint = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        x_val = torch.from_numpy(x_norm[val_idx]).to(device)
        pred_norm = model(x_val).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    target = y[val_idx]
    pred_reshaped = pred.reshape(len(pred), -1, 3)
    target_reshaped = target.reshape(len(target), -1, 3)
    xy_error = np.linalg.norm(pred_reshaped[:, :, :2] - target_reshaped[:, :, :2], axis=2)
    yaw_error = np.abs((pred_reshaped[:, :, 2] - target_reshaped[:, :, 2] + np.pi) % (2 * np.pi) - np.pi)

    metrics: Dict[str, object] = {
        "data": str(data_path),
        "device": str(device),
        "samples": int(len(x)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "split_mode": args.split_mode,
        "mean_xy_error_m_by_horizon": xy_error.mean(axis=0).tolist(),
        "mean_yaw_error_rad_by_horizon": yaw_error.mean(axis=0).tolist(),
        "final_horizon_xy_error_m": float(xy_error[:, -1].mean()),
    }

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "val_predictions.npz", pred=pred.astype(np.float32), target=target.astype(np.float32), val_idx=val_idx)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a route-following future-trajectory policy.")
    parser.add_argument("--data", required=True, help="Dataset .npz from route_dataset.")
    parser.add_argument("--out-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-mode", choices=["group_random", "random", "time"], default="group_random")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
