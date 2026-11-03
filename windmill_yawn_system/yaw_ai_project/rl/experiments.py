"""Ablation and multi-seed experiments for DQN / Double DQN.

Run examples (from the project root)::

    python rl/experiments.py --mode ablation --profile reduced --seeds 0
    python rl/experiments.py --mode multi_seed --profile reduced --seeds 0,1,2,3,4

The ``reduced`` profile exists because a full 2,200-episode x 1,000-step run per
seed is expensive on CPU.  Reduced runs use shorter episodes and a faster
epsilon schedule, but *all algorithms inside a comparison use exactly the same
settings*.  Results are labelled ``reduced budget`` in every CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.environment import YawRLEnvironment
from rl.agents import (
    DQNConfig,
    DQNAgent,
    DoubleDQNAgent,
    set_global_seed,
)
from rl.metrics import MetricLogger, confidence_interval_95
from rl.trainer import evaluate_policy, train_agent

# Hyperparameters that stay identical across every experimental variant.
SHARED_HYPERPARAMETERS: Dict[str, float] = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "grad_clip_norm": 5.0,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
}

PROFILES: Dict[str, Dict[str, int]] = {
    # Original project budget (2,200 episodes x 1,000 steps).
    "full": {
        "episodes": 2200,
        "steps_per_episode": 1000,
        "epsilon_decay_steps": 1500,
        "train_start_steps": 2000,
        "target_update_every": 250,
        "batch_size": 256,
        "buffer_size": 50_000,
    },
    # Faster budget used for reproducible smoke experiments on CPU.
    "reduced": {
        "episodes": 80,
        "steps_per_episode": 300,
        "epsilon_decay_steps": 60,
        "train_start_steps": 500,
        "target_update_every": 100,
        "batch_size": 128,
        "buffer_size": 20_000,
    },
}

ABLATION_VARIANTS = ("standard_dqn", "no_replay", "no_target_network", "double_dqn")
MULTI_SEED_ALGORITHMS = ("dqn", "double_dqn")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def build_config(
    name: str,
    profile: Dict[str, int],
    seed: int,
    device: str,
    use_replay: bool = True,
    use_target_network: bool = True,
    double_dqn: bool = False,
) -> DQNConfig:
    return DQNConfig(
        gamma=SHARED_HYPERPARAMETERS["gamma"],
        learning_rate=SHARED_HYPERPARAMETERS["learning_rate"],
        grad_clip_norm=SHARED_HYPERPARAMETERS["grad_clip_norm"],
        epsilon_start=SHARED_HYPERPARAMETERS["epsilon_start"],
        epsilon_end=SHARED_HYPERPARAMETERS["epsilon_end"],
        epsilon_decay_steps=int(profile["epsilon_decay_steps"]),
        train_start_steps=int(profile["train_start_steps"]),
        target_update_every=int(profile["target_update_every"]),
        batch_size=int(profile["batch_size"]),
        buffer_size=int(profile["buffer_size"]),
        use_replay=use_replay,
        use_target_network=use_target_network,
        double_dqn=double_dqn,
        device=device,
        seed=seed,
        name=name,
    )


def build_agent(name: str, profile: Dict[str, int], seed: int, device: str) -> DQNAgent:
    """Instantiate one of the four ablation variants."""

    if name == "standard_dqn":
        return DQNAgent(build_config(name, profile, seed, device))
    if name == "no_replay":
        return DQNAgent(
            build_config(name, profile, seed, device, use_replay=False, use_target_network=True)
        )
    if name == "no_target_network":
        return DQNAgent(
            build_config(name, profile, seed, device, use_replay=True, use_target_network=False)
        )
    if name == "double_dqn":
        return DoubleDQNAgent(build_config(name, profile, seed, device, double_dqn=True))
    if name == "dqn":
        return DQNAgent(build_config(name, profile, seed, device))
    raise ValueError(f"Unknown experiment variant: {name}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def training_diagnostics(metrics_path: Path) -> Dict[str, float]:
    """Summarise step-level stability from the real CSV log.

    ``diverged`` is intentionally conservative: it flags non-finite values or
    catastrophic magnitudes, not ordinary early-training loss growth.  The
    continuous ``loss_growth_ratio`` column is provided for inspection.
    """

    if not metrics_path.exists():
        return {
            "loss_std": float("nan"),
            "max_grad_norm": float("nan"),
            "loss_growth_ratio": float("nan"),
            "diverged": 0,
        }
    frame = pd.read_csv(metrics_path)
    if frame.empty or "loss" not in frame:
        return {
            "loss_std": float("nan"),
            "max_grad_norm": float("nan"),
            "loss_growth_ratio": float("nan"),
            "diverged": 0,
        }

    losses = frame["loss"].astype(float).to_numpy()
    finite = np.isfinite(losses)
    grad = frame.get("grad_norm", pd.Series(dtype=float)).astype(float).to_numpy()

    growth_ratio = float("nan")
    if len(losses) >= 20 and np.isfinite(losses[:10]).any() and np.isfinite(losses[-10:]).any():
        early = float(np.nanmedian(losses[:10]))
        late = float(np.nanmedian(losses[-10:]))
        if early > 1e-9:
            growth_ratio = late / early

    max_grad = (
        float(np.nanmax(grad)) if len(grad) and np.isfinite(grad).any() else float("nan")
    )
    max_loss = float(np.nanmax(np.abs(losses))) if finite.any() else float("nan")
    diverged = int(
        (not finite.all())
        or (np.isfinite(max_grad) and max_grad > 1e4)
        or (np.isfinite(max_loss) and max_loss > 1e6)
    )
    return {
        "loss_std": float(np.nanstd(losses)) if finite.any() else float("nan"),
        "max_grad_norm": max_grad,
        "loss_growth_ratio": growth_ratio,
        "diverged": diverged,
    }


def run_single(
    name: str,
    profile_name: str,
    profile: Dict[str, int],
    seed: int,
    device: str,
    data_path: str,
    training_dir: Path,
    models_dir: Path,
    eval_episodes: int,
    eval_seed: int,
    log_interval: int,
    print_interval: int,
) -> Dict[str, object]:
    """Train one agent, evaluate it, and return a flat result row."""

    run_name = f"{name}_seed{seed}"
    run_dir = training_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {run_name} ({profile_name}) ===")
    set_global_seed(seed)
    env = YawRLEnvironment(data_path=data_path)
    agent = build_agent(name, profile, seed, device)

    logger = MetricLogger(
        output_dir=run_dir,
        prefix=run_name,
    )
    try:
        train_summary = train_agent(
            env=env,
            agent=agent,
            episodes=int(profile["episodes"]),
            steps_per_episode=int(profile["steps_per_episode"]),
            logger=logger,
            log_interval=log_interval,
            print_interval=print_interval,
            seed=seed,
        )
    finally:
        logger.close()

    model_path = models_dir / f"{name}_seed{seed}.pth"
    agent.save(model_path)

    # A separate evaluation environment keeps training and evaluation traces
    # independent while still using the same simulator.
    eval_env = YawRLEnvironment(data_path=data_path)
    eval_results = evaluate_policy(
        eval_env,
        action_fn=lambda state: agent.select_action(state, greedy=True),
        episodes=eval_episodes,
        seed_base=eval_seed,
        measure_inference=True,
    )
    eval_frame = pd.DataFrame(eval_results)
    diagnostics = training_diagnostics(run_dir / f"{run_name}_metrics.csv")

    row: Dict[str, object] = {
        "algorithm": name,
        "seed": seed,
        "profile": profile_name,
        "episodes": int(profile["episodes"]),
        "steps_per_episode": int(profile["steps_per_episode"]),
        "total_steps": train_summary["total_steps"],
        "optimization_steps": train_summary["total_optimization_steps"],
        "wall_time_s": train_summary["wall_time_s"],
        "train_final_episode_reward": train_summary["final_episode_reward"],
        "train_final_avg_reward_50": train_summary["final_avg_reward_50"],
        "train_best_avg_reward_50": train_summary["best_avg_reward_50"],
        "train_mean_loss": train_summary["mean_loss"],
        "train_mean_grad_norm": train_summary["mean_grad_norm"],
        "loss_std": diagnostics["loss_std"],
        "max_grad_norm": diagnostics["max_grad_norm"],
        "loss_growth_ratio": diagnostics["loss_growth_ratio"],
        "diverged": diagnostics["diverged"],
        "model_path": str(model_path),
        "notes": "reduced budget" if profile_name == "reduced" else "",
    }
    metric_columns = [
        "episode_reward",
        "average_power_normalized",
        "average_power_physical",
        "mean_abs_misalignment",
        "yaw_movement_frequency",
        "control_stability",
        "time_within_10_deg",
        "cumulative_yaw_movement",
        "inference_time_ms",
    ]
    for column in metric_columns:
        if column not in eval_frame:
            continue
        values = eval_frame[column].astype(float).to_numpy()
        # Raw per-run value plus variability across evaluation episodes.
        # ``summarise_runs`` then adds the cross-seed ``_mean``/``_std``
        # suffixes without producing names such as ``..._mean_mean``.
        row[f"eval_{column}"] = float(np.mean(values))
        row[f"eval_{column}_episode_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        row[f"eval_{column}_episode_ci95"] = confidence_interval_95(values.tolist())

    print(
        f"{run_name}: train_avg50={row['train_final_avg_reward_50']:.3f}, "
        f"eval_reward={row['eval_episode_reward']:.3f}, "
        f"eval_power_norm={row['eval_average_power_normalized']:.3f}, "
        f"eval_power_W={row['eval_average_power_physical']:.1f}"
    )
    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise_runs(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate runs with mean, standard deviation and 95% CI."""

    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"seed", "episodes", "steps_per_episode"}
    value_columns = [column for column in numeric if column not in exclude]
    rows: List[Dict[str, object]] = []
    for group_key, group in frame.groupby(list(group_columns), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: Dict[str, object] = dict(zip(group_columns, group_key))
        row["n_runs"] = int(len(group))
        for column in value_columns:
            values = group[column].astype(float).to_numpy()
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_ci95"] = confidence_interval_95(values.tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def write_experiment_outputs(
    rows: List[Dict[str, object]],
    prefix: str,
    output_dir: Path,
    group_columns: Sequence[str],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    raw_path = output_dir / f"{prefix}_results.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    frame.to_csv(raw_path, index=False)
    summarise_runs(frame, group_columns).to_csv(summary_path, index=False)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    return raw_path, summary_path


def parse_seeds(text: str) -> List[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation and multi-seed DQN experiments")
    parser.add_argument("--mode", choices=["ablation", "multi_seed", "both"], default="ablation")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="reduced")
    parser.add_argument("--seeds", default="0", help="Comma-separated seeds")
    parser.add_argument("--algorithms", default=",".join(MULTI_SEED_ALGORITHMS))
    parser.add_argument("--episodes", type=int, default=None, help="Override profile episodes")
    parser.add_argument("--steps", type=int, default=None, help="Override profile steps per episode")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=str(Path("data") / "era5_chennai.csv"))
    parser.add_argument("--training-dir", default=str(Path("results") / "training"))
    parser.add_argument("--models-dir", default=str(Path("results") / "models"))
    parser.add_argument("--output-dir", default=str(Path("results") / "evaluation"))
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--print-interval", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Wind dataset not found: {args.data_path}")

    profile = dict(PROFILES[args.profile])
    if args.episodes is not None:
        profile["episodes"] = args.episodes
    if args.steps is not None:
        profile["steps_per_episode"] = args.steps

    seeds = parse_seeds(args.seeds)
    algorithms = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    training_dir = Path(args.training_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("ablation", "both"):
        rows = []
        for name in ABLATION_VARIANTS:
            for seed in seeds:
                rows.append(
                    run_single(
                        name=name,
                        profile_name=args.profile,
                        profile=profile,
                        seed=seed,
                        device=args.device,
                        data_path=args.data_path,
                        training_dir=training_dir,
                        models_dir=models_dir,
                        eval_episodes=args.eval_episodes,
                        eval_seed=args.eval_seed,
                        log_interval=args.log_interval,
                        print_interval=args.print_interval,
                    )
                )
        write_experiment_outputs(rows, "ablation", output_dir, ["algorithm"])

    if args.mode in ("multi_seed", "both"):
        unknown = [name for name in algorithms if name not in ("dqn", "double_dqn")]
        if unknown:
            raise ValueError(f"Unknown algorithms: {unknown}")
        rows = []
        for name in algorithms:
            for seed in seeds:
                rows.append(
                    run_single(
                        name=name,
                        profile_name=args.profile,
                        profile=profile,
                        seed=seed,
                        device=args.device,
                        data_path=args.data_path,
                        training_dir=training_dir,
                        models_dir=models_dir,
                        eval_episodes=args.eval_episodes,
                        eval_seed=args.eval_seed,
                        log_interval=args.log_interval,
                        print_interval=args.print_interval,
                    )
                )
        write_experiment_outputs(rows, "multi_seed", output_dir, ["algorithm"])

        # Deployment artifact: the first seed is copied to a stable canonical
        # name.  This avoids selecting on evaluation performance.
        if seeds:
            for name in algorithms:
                source = models_dir / f"{name}_seed{seeds[0]}.pth"
                if source.exists():
                    target = models_dir / f"{name}.pth"
                    target.write_bytes(source.read_bytes())
                    print(f"Canonical model: {target}")


if __name__ == "__main__":
    main()
