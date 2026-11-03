"""Generate training plots from the CSV metric files.

The module reads only real logged data and skips a figure gracefully if the
corresponding CSV is missing or empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None
    if frame.empty:
        print(f"Skipping empty file: {path}")
        return None
    return frame


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_episode_reward(episodes: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if episodes is None or "episode_reward" not in episodes:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(episodes["episode"], episodes["episode_reward"], alpha=0.25, label="Episode reward")
    ax.plot(episodes["episode"], episodes["moving_avg_50"], linewidth=2, label="Moving average (50)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"{label}: Training Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_loss(steps: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if steps is None or "loss" not in steps:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps["global_step"], steps["loss"], linewidth=0.8, alpha=0.55, label="TD loss")
    if len(steps) >= 20:
        smooth = steps["loss"].rolling(window=20, min_periods=1).mean()
        ax.plot(steps["global_step"], smooth, linewidth=2, label="Rolling mean (20)")
    ax.set_xlabel("Optimization step (global environment step)")
    ax.set_ylabel("MSE TD loss")
    ax.set_title(f"{label}: Temporal-Difference Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_epsilon(episodes: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if episodes is None or "epsilon" not in episodes:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes["episode"], episodes["epsilon"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"{label}: Exploration Schedule")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_q_values(steps: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if steps is None or "avg_q" not in steps:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps["global_step"], steps["avg_q"], linewidth=1.0, label="Mean chosen Q")
    ax.plot(steps["global_step"], steps["max_q"], linewidth=1.0, alpha=0.7, label="Max chosen Q")
    ax.set_xlabel("Optimization step (global environment step)")
    ax.set_ylabel("Q-value")
    ax.set_title(f"{label}: Q-value Progression")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_gradient_norm(steps: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if steps is None or "grad_norm" not in steps:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps["global_step"], steps["grad_norm"], linewidth=0.9, alpha=0.6, label="Gradient norm")
    if len(steps) >= 20:
        smooth = steps["grad_norm"].rolling(window=20, min_periods=1).mean()
        ax.plot(steps["global_step"], smooth, linewidth=2, label="Rolling mean (20)")
    ax.set_xlabel("Optimization step (global environment step)")
    ax.set_ylabel("L2 gradient norm (before clipping)")
    ax.set_title(f"{label}: Gradient Norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_episode_length(episodes: pd.DataFrame, path: Path, label: str = "DQN") -> None:
    if episodes is None or "episode_length" not in episodes:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes["episode"], episodes["episode_length"], linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title(f"{label}: Episode Length")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def plot_training_time(summary: pd.DataFrame, path: Path) -> None:
    if summary is None or "algorithm" not in summary or "wall_time_s" not in summary:
        return
    grouped = summary.groupby("algorithm", as_index=False)["wall_time_s"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(grouped["algorithm"].str.upper(), grouped["wall_time_s"], alpha=0.85)
    for index, value in enumerate(grouped["wall_time_s"]):
        ax.text(index, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wall-clock training time (s)")
    ax.set_title("Training Time Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)


def plot_algorithm_comparison(
    dqn_steps: Optional[pd.DataFrame],
    double_steps: Optional[pd.DataFrame],
    dqn_episodes: Optional[pd.DataFrame],
    double_episodes: Optional[pd.DataFrame],
    path: Path,
) -> None:
    if dqn_episodes is None and double_episodes is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    for frame, label in ((dqn_episodes, "DQN"), (double_episodes, "Double DQN")):
        if frame is not None and "moving_avg_50" in frame:
            axes[0].plot(frame["episode"], frame["moving_avg_50"], linewidth=2, label=f"{label} avg(50)")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("DQN vs Double DQN: Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for frame, label in ((dqn_steps, "DQN"), (double_steps, "Double DQN")):
        if frame is not None and "loss" in frame:
            smooth = frame["loss"].rolling(window=20, min_periods=1).mean()
            axes[1].plot(frame["global_step"], smooth, linewidth=2, label=f"{label} loss (rolling 20)")
    axes[1].set_xlabel("Global environment step")
    axes[1].set_ylabel("MSE loss")
    axes[1].set_title("DQN vs Double DQN: TD Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    _save(fig, path)


def plot_comparison_summary(
    summary: Optional[pd.DataFrame],
    path: Path,
    title: str,
    label_column: str = "algorithm",
) -> None:
    """Bar charts with error bars from an experiment summary CSV."""

    if summary is None or label_column not in summary:
        return

    specs = (
        ("eval_episode_reward", "Evaluation reward"),
        ("eval_average_power_normalized", "Normalized power"),
        ("eval_average_power_physical", "Physical power (W)"),
        ("eval_control_stability", "Control stability (std yaw change)"),
    )
    available = [
        (prefix, label)
        for prefix, label in specs
        if f"{prefix}_mean" in summary.columns
    ]
    if not available:
        print(f"Skipping {title}: no recognised metric columns")
        return

    columns = 2
    rows = int(np.ceil(len(available) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 4.2 * rows), squeeze=False)
    labels = summary[label_column].astype(str).tolist()
    x = np.arange(len(labels))

    for axis, (prefix, label) in zip(axes.ravel(), available):
        means = summary[f"{prefix}_mean"].astype(float).to_numpy()
        stds = (
            summary[f"{prefix}_std"].astype(float).to_numpy()
            if f"{prefix}_std" in summary.columns
            else np.zeros_like(means)
        )
        axis.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(True, axis="y", alpha=0.3)

    for axis in axes.ravel()[len(available):]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_controller_comparison(summary: Optional[pd.DataFrame], path: Path) -> None:
    """Compare Random/Q-learning/DQN/Double DQN/Rule/PID on shared traces."""

    if summary is None or "controller" not in summary:
        return

    specs = (
        ("episode_reward", "Evaluation reward"),
        ("average_power_normalized", "Normalized power"),
        ("average_power_physical", "Physical power (W)"),
        ("mean_abs_misalignment", "Mean abs misalignment (deg)"),
        ("yaw_movement_frequency", "Yaw movement frequency"),
        ("time_within_10_deg", "Time within 10 deg"),
    )
    available = [
        (prefix, label)
        for prefix, label in specs
        if f"{prefix}_mean" in summary.columns
    ]
    if not available:
        print("Skipping controller comparison: no recognised metric columns")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    labels = summary["controller"].astype(str).tolist()
    x = np.arange(len(labels))
    for axis, (prefix, label) in zip(axes.ravel(), available):
        mean_column = f"{prefix}_mean"
        std_column = f"{prefix}_std"
        means = summary[mean_column].astype(float).to_numpy()
        stds = (
            summary[std_column].astype(float).to_numpy()
            if std_column in summary.columns
            else np.zeros_like(means)
        )
        axis.bar(x, means, yerr=stds, capsize=3, alpha=0.85)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_title(label)
        axis.set_ylabel(label)
        axis.grid(True, axis="y", alpha=0.3)
    for axis in axes.ravel()[len(available):]:
        axis.axis("off")
    fig.suptitle("Controller evaluation on identical wind traces")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DQN/Double DQN training metrics")
    parser.add_argument("--metrics-dir", default="results/training")
    parser.add_argument("--plots-dir", default="results/plots")
    parser.add_argument("--evaluation-dir", default="results/evaluation")
    parser.add_argument("--prefix", default="dqn")
    parser.add_argument("--double-prefix", default="double_dqn")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    plots_dir = Path(args.plots_dir)
    evaluation_dir = Path(args.evaluation_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = _read(metrics_dir / f"{args.prefix}_metrics.csv")
    episodes = _read(metrics_dir / f"{args.prefix}_episodes.csv")
    double_steps = _read(metrics_dir / f"{args.double_prefix}_metrics.csv")
    double_episodes = _read(metrics_dir / f"{args.double_prefix}_episodes.csv")
    summary = _read(metrics_dir / "training_summary.csv")

    plot_episode_reward(episodes, plots_dir / "dqn_reward.png", label=args.prefix.upper())
    plot_loss(steps, plots_dir / "dqn_loss.png", label=args.prefix.upper())
    plot_epsilon(episodes, plots_dir / "dqn_epsilon.png", label=args.prefix.upper())
    plot_q_values(steps, plots_dir / "dqn_q_values.png", label=args.prefix.upper())
    plot_gradient_norm(steps, plots_dir / "dqn_gradient_norm.png", label=args.prefix.upper())
    plot_episode_length(episodes, plots_dir / "dqn_episode_length.png", label=args.prefix.upper())
    plot_training_time(summary, plots_dir / "training_time_comparison.png")
    plot_algorithm_comparison(
        steps,
        double_steps,
        episodes,
        double_episodes,
        plots_dir / "dqn_vs_double_dqn.png",
    )

    ablation = _read(evaluation_dir / "ablation_summary.csv")
    if ablation is None:
        ablation = _read(evaluation_dir / "ablation_results.csv")
    plot_comparison_summary(
        ablation,
        plots_dir / "ablation_comparison.png",
        title="Ablation Study (reduced budget where indicated)",
        label_column="algorithm",
    )

    multi_seed = _read(evaluation_dir / "multi_seed_summary.csv")
    if multi_seed is None:
        multi_seed = _read(evaluation_dir / "multi_seed_results.csv")
    plot_comparison_summary(
        multi_seed,
        plots_dir / "multi_seed_comparison.png",
        title="Multi-Seed Evaluation (DQN vs Double DQN)",
        label_column="algorithm",
    )

    controller_summary = _read(evaluation_dir / "controller_comparison_summary.csv")
    plot_controller_comparison(controller_summary, plots_dir / "controller_comparison.png")


if __name__ == "__main__":
    main()
