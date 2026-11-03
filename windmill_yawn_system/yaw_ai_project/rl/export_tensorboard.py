"""Export existing CSV training logs to TensorBoard event files.

This is useful when training was executed without ``--tensorboard-dir`` or when
TensorBoard was not installed at training time.  The CSV files remain the source
of truth; this script only mirrors their real values into event files.

Example::

    python rl/export_tensorboard.py \
        --metrics-dir results/training \
        --tensorboard-dir results/tensorboard \
        --prefixes dqn,double_dqn
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - availability differs by environment
    from torch.utils.tensorboard import SummaryWriter
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "TensorBoard export requires torch.utils.tensorboard and tensorboard. "
        f"Import failed with: {exc}"
    )


STEP_SCALARS = (
    "loss",
    "grad_norm",
    "avg_q",
    "max_q",
    "epsilon",
    "learning_rate",
)

EPISODE_SCALARS = (
    "episode_reward",
    "moving_avg_50",
    "epsilon",
    "episode_length",
    "mean_loss",
    "mean_grad_norm",
    "mean_q",
    "max_q",
    "learning_rate",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def export_prefix(metrics_dir: Path, tensorboard_dir: Path, prefix: str) -> int:
    """Write one run's step/episode CSVs to a TensorBoard event directory."""

    steps_path = metrics_dir / f"{prefix}_metrics.csv"
    episodes_path = metrics_dir / f"{prefix}_episodes.csv"
    if not steps_path.exists() and not episodes_path.exists():
        print(f"[skip] No metrics found for prefix '{prefix}' in {metrics_dir}")
        return 0

    writer = SummaryWriter(log_dir=str(tensorboard_dir / prefix))
    written = 0
    try:
        if steps_path.exists():
            steps = pd.read_csv(steps_path)
            for column in STEP_SCALARS:
                if column not in steps:
                    continue
                for global_step, value in zip(steps["global_step"], steps[column]):
                    if _finite(value):
                        writer.add_scalar(f"train/{column}", float(value), int(global_step))
                        written += 1

        if episodes_path.exists():
            episodes = pd.read_csv(episodes_path)
            for column in EPISODE_SCALARS:
                if column not in episodes:
                    continue
                for episode, value in zip(episodes["episode"], episodes[column]):
                    if _finite(value):
                        writer.add_scalar(f"episode/{column}", float(value), int(episode))
                        written += 1
    finally:
        writer.flush()
        writer.close()

    print(f"[ok] {prefix}: {written} scalar points -> {tensorboard_dir / prefix}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CSV training metrics to TensorBoard")
    parser.add_argument("--metrics-dir", default=str(Path("results") / "training"))
    parser.add_argument("--tensorboard-dir", default=str(Path("results") / "tensorboard"))
    parser.add_argument("--prefixes", default="dqn,double_dqn")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    tensorboard_dir = Path(args.tensorboard_dir)
    prefixes: Iterable[str] = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    total = 0
    for prefix in prefixes:
        total += export_prefix(metrics_dir, tensorboard_dir, prefix)

    if total == 0:
        raise SystemExit("Nothing exported. Check --metrics-dir and --prefixes.")
    print(f"\nLaunch with: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()
