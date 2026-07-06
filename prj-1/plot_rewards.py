"""Plot episode rewards from Stable-Baselines3 Monitor logs.

The training script writes Gymnasium episode statistics through SB3's Monitor wrapper.
This script reads the monitor.csv file, plots raw episode reward, and overlays a
simple moving average so the learning trend is easier to see.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

DEFAULT_LOG_DIR = Path("logs/monitor")
DEFAULT_OUTPUT_PATH = Path("plots/reward_curve.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward curve from SB3 monitor logs")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory containing monitor.csv.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the plotted image.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Moving-average window used to smooth the reward curve.",
    )
    return parser.parse_args()


def read_monitor_rewards(log_dir: Path) -> list[float]:
    monitor_files = sorted(log_dir.rglob("monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No monitor.csv files found under {log_dir}")

    rewards: list[float] = []
    for monitor_file in monitor_files:
        with monitor_file.open("r", newline="") as handle:
            header: list[str] | None = None
            for raw_line in handle:
                if raw_line.startswith("#"):
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if header is None:
                    header = line.split(",")
                    continue
                values = line.split(",")
                row = dict(zip(header, values))
                rewards.append(float(row["r"]))
    return rewards


def moving_average(values: Iterable[float], window_size: int) -> list[float]:
    if window_size <= 1:
        return list(values)

    window = deque(maxlen=window_size)
    smoothed: list[float] = []
    for value in values:
        window.append(value)
        smoothed.append(sum(window) / len(window))
    return smoothed


def main() -> None:
    args = parse_args()
    rewards = read_monitor_rewards(args.log_dir)
    smoothed = moving_average(rewards, args.window_size)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = list(range(1, len(rewards) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    plt.plot(episodes, smoothed, linewidth=2.0, label=f"{args.window_size}-episode moving average")
    plt.title("CartPole-v1 Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_path, dpi=200)
    print(f"Saved reward plot to {args.output_path}")


if __name__ == "__main__":
    main()
