"""Evaluate every controller on identical wind conditions.

The controllers compared here are:

* Random policy
* Tabular Q-learning
* Standard DQN
* Double DQN
* Rule-based controller
* PID controller

All controllers act through the same :class:`YawRLEnvironment` and therefore
experience identical wind traces, actuator delay, inertia and rate limits.
Rule and PID normally emit continuous yaw commands; for a fair comparison with
the three discrete RL actions their command is projected onto the same
``left / right / hold`` action set.  The original continuous Rule/PID results in
``results/controller_comparison_summary.csv`` are preserved separately.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.environment import YawRLEnvironment
from rl.agents import (
    ACTION_SIZE,
    Q_BINS,
    DQNAgent,
    DQNConfig,
    discretize_state,
    infer_hidden_sizes,
    set_global_seed,
    train_tabular_q_learning,
)
from rl.metrics import confidence_interval_95
from rl.trainer import evaluate_policy

import torch

RULE_THRESHOLD_DEG = 10.0
PID_KP = 0.5
PID_KD = 0.1
PID_MAX_DELTA = 2.0
# A command smaller than half an action step is projected to ``hold``.
ACTION_DEADBAND_DEG = 0.5

DEFAULT_EPISODES = 10
DEFAULT_EVAL_SEED = 10_000
DEFAULT_OUTPUT_DIR = Path("results") / "evaluation"
DEFAULT_MODEL_DIR = Path("results") / "models"


# ---------------------------------------------------------------------------
# State helpers (the policy only sees the 5-D observation)
# ---------------------------------------------------------------------------

def misalignment_from_state(state: np.ndarray) -> float:
    """Recover the signed misalignment in degrees from the normalized state."""

    return float(state[3]) * 360.0 - 180.0


def wind_change_from_state(state: np.ndarray) -> float:
    """Recover the wind-direction change in degrees from the normalized state."""

    return float(state[4]) * 360.0 - 180.0


def command_to_action(command_deg: float) -> int:
    """Project a continuous yaw command onto the discrete action set."""

    if command_deg > ACTION_DEADBAND_DEG:
        return 1  # right
    if command_deg < -ACTION_DEADBAND_DEG:
        return 0  # left
    return 2  # hold


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

def rule_action(state: np.ndarray) -> int:
    """Classic dead-band rule controller projected onto discrete actions."""

    misalignment = misalignment_from_state(state)
    if abs(misalignment) > RULE_THRESHOLD_DEG:
        return command_to_action(np.sign(misalignment) * 2.0)
    return 2


def pid_action(state: np.ndarray) -> int:
    """PD/PID-style controller projected onto discrete actions."""

    misalignment = misalignment_from_state(state)
    wind_change = wind_change_from_state(state)
    command = PID_KP * misalignment + PID_KD * wind_change
    command = float(np.clip(command, -PID_MAX_DELTA, PID_MAX_DELTA))
    return command_to_action(command)


def make_random_action(seed: int) -> Callable[[np.ndarray], int]:
    rng = np.random.default_rng(seed)
    return lambda _state: int(rng.integers(0, ACTION_SIZE))


def make_q_learning_action(q_table: np.ndarray) -> Callable[[np.ndarray], int]:
    def policy(state: np.ndarray) -> int:
        return int(np.argmax(q_table[discretize_state(state, Q_BINS)]))

    return policy


def make_agent_action(agent: DQNAgent) -> Callable[[np.ndarray], int]:
    def policy(state: np.ndarray) -> int:
        return agent.select_action(state, greedy=True)

    return policy


# ---------------------------------------------------------------------------
# Loading / training artifacts
# ---------------------------------------------------------------------------

def load_agent(path: str | Path, device: str = "auto") -> Optional[DQNAgent]:
    """Load a DQN checkpoint, returning ``None`` if the file is missing."""

    path = Path(path)
    if not path.exists():
        print(f"[skip] Checkpoint not found: {path}")
        return None
    try:
        agent = DQNAgent.from_checkpoint(path, device=device)
        print(f"[load] {path} -> {agent.config.name} on {agent.device}")
        return agent
    except Exception as exc:  # pragma: no cover - defensive for corrupt files
        print(f"[skip] Could not load {path}: {exc}")
        return None


def load_checkpoint_metadata(path: str | Path) -> Dict[str, object]:
    """Return architecture metadata without constructing an agent."""

    path = Path(path)
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", {})
    return {
        "hidden_sizes": list(infer_hidden_sizes(state_dict)),
        "algorithm": checkpoint.get("algorithm", "unknown"),
    }


def load_or_train_q_table(
    env: YawRLEnvironment,
    path: str | Path,
    episodes: int,
    steps_per_episode: int,
    seed: int,
) -> Optional[np.ndarray]:
    """Load a saved Q-table, or train one when requested."""

    path = Path(path)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        q_table = data["q_table"]
        print(f"[load] Q-table {path} shape={q_table.shape}")
        return q_table
    if episodes <= 0:
        print("[skip] Q-table not found and --q-episodes is 0")
        return None

    print(f"[train] Tabular Q-learning for {episodes} episodes ...")
    q_table, _rewards, _history = train_tabular_q_learning(
        env,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        seed=seed,
        log_interval=max(1, episodes // 5),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q_table=q_table)
    print(f"[save] Q-table {path}")
    return q_table


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

def per_controller_summary(rows: List[Dict[str, float]], controller: str) -> Dict[str, object]:
    frame = pd.DataFrame(rows)
    summary: Dict[str, object] = {"controller": controller, "episodes": len(frame)}
    for column in (
        "episode_reward",
        "average_power_normalized",
        "average_power_physical",
        "total_energy_normalized",
        "total_energy_physical",
        "mean_abs_misalignment",
        "yaw_movement_frequency",
        "control_stability",
        "cumulative_yaw_movement",
        "yaw_reversals",
        "time_within_10_deg",
        "inference_time_ms",
    ):
        if column not in frame:
            continue
        values = frame[column].astype(float).to_numpy()
        summary[f"{column}_mean"] = float(np.mean(values))
        summary[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"{column}_ci95"] = confidence_interval_95(values.tolist())
    return summary


def evaluate_controllers(
    controllers: Dict[str, Callable[[np.ndarray], int]],
    episodes: int,
    eval_seed: int,
    data_path: str | Path,
    dt: float = 1.0,
) -> tuple[List[Dict[str, float]], List[Dict[str, object]]]:
    """Run every controller on the same evaluation seeds."""

    env = YawRLEnvironment(data_path=str(data_path))
    raw_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, object]] = []

    for name, policy in controllers.items():
        print(f"[eval] {name}: {episodes} episodes")
        results = evaluate_policy(
            env,
            action_fn=policy,
            episodes=episodes,
            seed_base=eval_seed,
            dt=dt,
            measure_inference=name not in ("random", "rule", "pid"),
        )
        for index, metrics in enumerate(results):
            raw_rows.append({"controller": name, "episode": index + 1, **metrics})
        summary_rows.append(per_controller_summary(results, name))

    return raw_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all yaw controllers on identical wind conditions")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--steps", type=int, default=1000, help="Q-learning training episode length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=str(Path("data") / "era5_chennai.csv"))
    parser.add_argument("--dqn-model", default=str(Path("rl") / "model.pth"))
    parser.add_argument(
        "--double-dqn-model",
        default=str(Path("results") / "models" / "double_dqn.pth"),
    )
    parser.add_argument("--q-table", default=str(Path("results") / "models" / "q_table.npz"))
    parser.add_argument("--q-episodes", type=int, default=0, help="Train a Q-table if none exists")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--controllers",
        default="random,q_learning,dqn,double_dqn,rule,pid",
        help="Comma-separated subset of controllers",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Wind dataset not found: {args.data_path}")

    requested = [name.strip() for name in args.controllers.split(",") if name.strip()]
    policies: Dict[str, Callable[[np.ndarray], int]] = {}

    if "random" in requested:
        policies["random"] = make_random_action(args.seed)

    if "rule" in requested:
        policies["rule"] = rule_action

    if "pid" in requested:
        policies["pid"] = pid_action

    env = YawRLEnvironment(data_path=args.data_path)

    if "q_learning" in requested:
        q_table = load_or_train_q_table(
            env,
            args.q_table,
            episodes=args.q_episodes,
            steps_per_episode=args.steps,
            seed=args.seed,
        )
        if q_table is not None:
            policies["q_learning"] = make_q_learning_action(q_table)

    if "dqn" in requested:
        agent = load_agent(args.dqn_model, device=args.device)
        if agent is not None:
            policies["dqn"] = make_agent_action(agent)

    if "double_dqn" in requested:
        agent = load_agent(args.double_dqn_model, device=args.device)
        if agent is not None:
            policies["double_dqn"] = make_agent_action(agent)

    if not policies:
        raise RuntimeError("No controller could be evaluated. Check paths and controller names.")

    # Preserve the user-facing order given on the command line.
    policies = {name: policies[name] for name in requested if name in policies}

    print(f"Evaluating controllers: {', '.join(policies)}")
    raw_rows, summary_rows = evaluate_controllers(
        policies,
        episodes=args.episodes,
        eval_seed=args.eval_seed,
        data_path=args.data_path,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "controller_comparison.csv"
    summary_path = output_dir / "controller_comparison_summary.csv"
    pd.DataFrame(raw_rows).to_csv(raw_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nSaved per-episode metrics: {raw_path}")
    print(f"Saved summary: {summary_path}\n")

    summary_frame = pd.DataFrame(summary_rows)
    display = summary_frame[
        [
            "controller",
            "episode_reward_mean",
            "average_power_normalized_mean",
            "average_power_physical_mean",
            "mean_abs_misalignment_mean",
            "yaw_movement_frequency_mean",
            "time_within_10_deg_mean",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
