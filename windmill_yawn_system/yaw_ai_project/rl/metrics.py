"""Metric logging utilities for DQN training.

CSV logging is always available.  TensorBoard logging is optional and is
enabled by passing ``tensorboard_dir``.  The two paths are independent, so the
original CSV workflow continues to work even if TensorBoard is not installed.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

STEP_FIELDS = [
    "episode",
    "global_step",
    "loss",
    "grad_norm",
    "avg_q",
    "max_q",
    "epsilon",
    "learning_rate",
]

EPISODE_FIELDS = [
    "episode",
    "episode_reward",
    "moving_avg_50",
    "epsilon",
    "episode_length",
    "episode_time_s",
    "total_time_s",
    "optimization_steps",
    "mean_loss",
    "mean_grad_norm",
    "mean_q",
    "max_q",
    "learning_rate",
]

SUMMARY_FIELDS = [
    "algorithm",
    "run_name",
    "seed",
    "episodes",
    "steps_per_episode",
    "total_steps",
    "total_optimization_steps",
    "wall_time_s",
    "final_episode_reward",
    "final_avg_reward_50",
    "best_avg_reward_50",
    "mean_loss",
    "mean_grad_norm",
    "total_trainable_parameters",
    "device",
    "timestamp",
    "notes",
]


class MetricLogger:
    """Write step-level, episode-level and run-level metrics to CSV."""

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str,
        summary_path: str | Path | None = None,
        tensorboard_dir: str | Path | None = None,
        tensorboard_run_name: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.step_path = self.output_dir / f"{prefix}_metrics.csv"
        self.episode_path = self.output_dir / f"{prefix}_episodes.csv"
        self.summary_path = Path(summary_path) if summary_path else self.output_dir / "training_summary.csv"

        self._step_file = open(self.step_path, "w", newline="", encoding="utf-8")
        self._step_writer = csv.DictWriter(self._step_file, fieldnames=STEP_FIELDS)
        self._step_writer.writeheader()

        self._episode_file = open(self.episode_path, "w", newline="", encoding="utf-8")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=EPISODE_FIELDS)
        self._episode_writer.writeheader()

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_exists = self.summary_path.exists()

        self.tb = None
        if tensorboard_dir is not None:
            # TensorBoard can import a sizeable optional dependency stack.
            # Keep ordinary CSV-only training lightweight by importing it only
            # when a caller explicitly asks for event-file logging.
            try:  # pragma: no cover - availability differs by environment
                from torch.utils.tensorboard import SummaryWriter
            except Exception:
                print("TensorBoard requested but torch.utils.tensorboard is unavailable; CSV logging only.")
            else:
                run_name = tensorboard_run_name or prefix
                self.tb = SummaryWriter(log_dir=str(Path(tensorboard_dir) / run_name))

    def log_step(
        self,
        episode: int,
        global_step: int,
        loss: float,
        grad_norm: float,
        avg_q: float,
        max_q: float,
        epsilon: float,
        learning_rate: float,
    ) -> None:
        row = {
            "episode": int(episode),
            "global_step": int(global_step),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "avg_q": float(avg_q),
            "max_q": float(max_q),
            "epsilon": float(epsilon),
            "learning_rate": float(learning_rate),
        }
        self._step_writer.writerow(row)
        if self.tb is not None:
            self.tb.add_scalar("train/loss", row["loss"], global_step)
            self.tb.add_scalar("train/grad_norm", row["grad_norm"], global_step)
            self.tb.add_scalar("train/avg_q", row["avg_q"], global_step)
            self.tb.add_scalar("train/max_q", row["max_q"], global_step)
            self.tb.add_scalar("train/epsilon", row["epsilon"], global_step)
            self.tb.add_scalar("train/learning_rate", row["learning_rate"], global_step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        moving_avg_50: float,
        epsilon: float,
        episode_length: int,
        episode_time_s: float,
        total_time_s: float,
        optimization_steps: int,
        mean_loss: float,
        mean_grad_norm: float,
        mean_q: float,
        max_q: float,
        learning_rate: float,
    ) -> None:
        row = {
            "episode": int(episode),
            "episode_reward": float(episode_reward),
            "moving_avg_50": float(moving_avg_50),
            "epsilon": float(epsilon),
            "episode_length": int(episode_length),
            "episode_time_s": float(episode_time_s),
            "total_time_s": float(total_time_s),
            "optimization_steps": int(optimization_steps),
            "mean_loss": float(mean_loss),
            "mean_grad_norm": float(mean_grad_norm),
            "mean_q": float(mean_q),
            "max_q": float(max_q),
            "learning_rate": float(learning_rate),
        }
        self._episode_writer.writerow(row)
        if self.tb is not None:
            self.tb.add_scalar("episode/reward", row["episode_reward"], episode)
            self.tb.add_scalar("episode/reward_avg50", row["moving_avg_50"], episode)
            self.tb.add_scalar("episode/epsilon", row["epsilon"], episode)
            self.tb.add_scalar("episode/length", row["episode_length"], episode)
            self.tb.add_scalar("episode/mean_loss", row["mean_loss"], episode)
            self.tb.add_scalar("episode/mean_grad_norm", row["mean_grad_norm"], episode)
            self.tb.add_scalar("episode/mean_q", row["mean_q"], episode)
            self.tb.add_scalar("episode/max_q", row["max_q"], episode)

    def log_summary(self, row: Dict[str, object]) -> None:
        full = {field: row.get(field, "") for field in SUMMARY_FIELDS}
        write_header = not self._summary_exists
        with open(self.summary_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(full)
        self._summary_exists = True
        if self.tb is not None:
            for key in ("final_avg_reward_50", "best_avg_reward_50", "wall_time_s"):
                value = row.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    self.tb.add_scalar(f"summary/{key}", float(value), 0)

    def flush(self) -> None:
        self._step_file.flush()
        self._episode_file.flush()

    def close(self) -> None:
        self._step_file.close()
        self._episode_file.close()
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def confidence_interval_95(values: Sequence[float]) -> float:
    """Return the half-width of a 95% t-based confidence interval.

    For small samples we use the normal approximation ``1.96 * s / sqrt(n)``.
    This is reported as an approximate interval and is clearly labelled as
    such in the generated tables.
    """

    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(clean) < 2:
        return float("nan")
    return float(1.96 * np.std(clean, ddof=1) / math.sqrt(len(clean)))


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "ci95": confidence_interval_95(clean),
        "n": len(clean),
    }
