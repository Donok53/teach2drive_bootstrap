import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .token_dataset import STOP_REASON_NAMES, STOP_STATE_NAMES


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def _confusion_matrix(target: np.ndarray, pred: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for target_item, pred_item in zip(target.astype(np.int64), pred.astype(np.int64)):
        if 0 <= target_item < class_count and 0 <= pred_item < class_count:
            matrix[target_item, pred_item] += 1
    return matrix


def _class_report(target: np.ndarray, pred: np.ndarray, names: List[str]) -> Dict:
    matrix = _confusion_matrix(target, pred, len(names))
    rows = {}
    for idx, name in enumerate(names):
        tp = int(matrix[idx, idx])
        support = int(matrix[idx].sum())
        predicted = int(matrix[:, idx].sum())
        precision = _safe_div(tp, predicted)
        recall = _safe_div(tp, support)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows[name] = {
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    accuracy = _safe_div(float(np.trace(matrix)), float(matrix.sum()))
    macro_f1 = float(np.mean([row["f1"] for row in rows.values()])) if rows else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classes": rows,
        "confusion_matrix": matrix.tolist(),
    }


def _binary_report(target: np.ndarray, prob: np.ndarray, threshold: float) -> Dict:
    target_bool = target.reshape(-1) >= 0.5
    pred_bool = prob.reshape(-1) >= threshold
    tp = int(np.sum(pred_bool & target_bool))
    tn = int(np.sum(~pred_bool & ~target_bool))
    fp = int(np.sum(pred_bool & ~target_bool))
    fn = int(np.sum(~pred_bool & target_bool))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _print_table(title: str, report: Dict) -> None:
    print(f"\n## {title}")
    print(f"accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f}")
    print("| class | support | predicted | precision | recall | f1 |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, row in report["classes"].items():
        print(
            f"| {name} | {row['support']} | {row['predicted']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )


def build_report(args: argparse.Namespace) -> Dict:
    run_dir = Path(args.run_dir).expanduser()
    metrics = _load_json(run_dir / "token_metrics.json")
    pred_path = run_dir / "token_val_predictions.npz"
    report = {
        "run_dir": str(run_dir),
        "metrics": metrics,
    }
    if not pred_path.exists():
        report["warning"] = f"Missing validation predictions: {pred_path}. Class-level reports require token_val_predictions.npz."
        return report

    arrays = np.load(pred_path)
    if "stop_prob" in arrays and "stop_target" in arrays:
        report["stop_binary"] = _binary_report(arrays["stop_target"], arrays["stop_prob"], args.stop_threshold)
    if "stop_state_pred" in arrays and "stop_state_target" in arrays:
        report["stop_state"] = _class_report(arrays["stop_state_target"], arrays["stop_state_pred"], STOP_STATE_NAMES)
    if "stop_reason_pred" in arrays and "stop_reason_target" in arrays and "stop_reason_mask" in arrays:
        mask = arrays["stop_reason_mask"].reshape(-1).astype(bool)
        report["stop_reason_label_ratio_val"] = float(np.mean(mask))
        if np.any(mask):
            report["stop_reason"] = _class_report(
                arrays["stop_reason_target"][mask],
                arrays["stop_reason_pred"][mask],
                STOP_REASON_NAMES,
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Print detailed validation metrics for a Teach2Drive token run.")
    parser.add_argument("--run-dir", required=True, help="Training output directory containing token_metrics.json and token_val_predictions.npz.")
    parser.add_argument("--output-json", default="", help="Optional path to save the detailed report JSON.")
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    args = parser.parse_args()

    report = build_report(args)
    metrics = report.get("metrics", {})
    print("# Teach2Drive Token Result Report")
    if "warning" in report:
        print(f"warning: {report['warning']}")
    for key in (
        "best_epoch",
        "best_val_loss",
        "final_horizon_xy_error_m",
        "stop_accuracy",
        "stop_state_accuracy",
        "stop_reason_accuracy",
        "stop_reason_label_ratio",
    ):
        if key in metrics:
            value = metrics[key]
            print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
    if "mean_xy_error_m_by_horizon" in metrics:
        print("mean_xy_error_m_by_horizon:", ", ".join(f"{item:.3f}" for item in metrics["mean_xy_error_m_by_horizon"]))
    if "mean_speed_error_mps_by_horizon" in metrics:
        print("mean_speed_error_mps_by_horizon:", ", ".join(f"{item:.3f}" for item in metrics["mean_speed_error_mps_by_horizon"]))
    if "stop_binary" in report:
        stop = report["stop_binary"]
        print(
            "\n## Stop Binary\n"
            f"accuracy={stop['accuracy']:.4f} precision={stop['precision']:.4f} "
            f"recall={stop['recall']:.4f} f1={stop['f1']:.4f} "
            f"tp={stop['tp']} fp={stop['fp']} fn={stop['fn']} tn={stop['tn']}"
        )
    if "stop_state" in report:
        _print_table("Stop State", report["stop_state"])
    if "stop_reason" in report:
        print(f"\nstop_reason_label_ratio_val={report['stop_reason_label_ratio_val']:.4f}")
        _print_table("Stop Reason", report["stop_reason"])

    if args.output_json:
        out = Path(args.output_json).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON report: {out}")


if __name__ == "__main__":
    main()
