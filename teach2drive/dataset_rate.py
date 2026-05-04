import argparse
import time
from pathlib import Path


def _directory_size(path):
    total = 0
    files = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
                files += 1
            except FileNotFoundError:
                pass
    return total, files


def _count_frames(path):
    total = 0
    for frames in path.glob("episode_*/frames.jsonl"):
        try:
            with frames.open("r", encoding="utf-8") as handle:
                total += sum(1 for _ in handle)
        except FileNotFoundError:
            pass
    return total


def _fmt_bytes(value):
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024


def monitor(args):
    path = Path(args.path).expanduser()
    last_size = None
    last_time = None
    while True:
        now = time.monotonic()
        size, files = _directory_size(path)
        frames = _count_frames(path)
        if last_size is None:
            rate = 0.0
        else:
            rate = (size - last_size) / max(now - last_time, 1e-6)
        gb_per_hour = rate * 3600 / (1024 ** 3)
        print(
            f"path={path} size={_fmt_bytes(size)} files={files} frames={frames} "
            f"rate={_fmt_bytes(rate)}/s est={gb_per_hour:.2f}GiB/h",
            flush=True,
        )
        last_size = size
        last_time = now
        time.sleep(args.interval_sec)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Monitor Teach2Drive tokenized dataset write rate.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--interval-sec", type=float, default=10.0)
    return parser


def main():
    monitor(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
