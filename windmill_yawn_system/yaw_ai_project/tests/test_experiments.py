"""Tests for experiment aggregation, evaluation helpers and diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.agent_evaluation import (
    ACTION_DEADBAND_DEG,
    command_to_action,
    pid_action,
    rule_action,
)
from env.environment import YawRLEnvironment
from rl.agents import DQNConfig, DQNAgent, DoubleDQNAgent, set_global_seed
from rl.experiments import summarise_runs, training_diagnostics
from rl.metrics import MetricLogger
from rl.trainer import evaluate_policy, train_agent


# ---------------------------------------------------------------------------
# Experiment aggregation
# ---------------------------------------------------------------------------

def test_summarise_runs_uses_clean_metric_names():
    frame = pd.DataFrame(
        {
            "algorithm": ["dqn", "dqn", "double_dqn"],
            "seed": [0, 1, 0],
            "eval_episode_reward": [1.0, 3.0, 2.0],
            "eval_average_power_normalized": [10.0, 12.0, 11.0],
        }
    )
    summary = summarise_runs(frame, ["algorithm"])

    assert "eval_episode_reward_mean" in summary
    assert "eval_episode_reward_std" in summary
    assert "eval_episode_reward_mean_mean" not in summary

    dqn_row = summary.loc[summary["algorithm"] == "dqn"].iloc[0]
    assert dqn_row["n_runs"] == 2
    assert dqn_row["eval_episode_reward_mean"] == pytest.approx(2.0)
    assert dqn_row["eval_episode_reward_std"] == pytest.approx(math.sqrt(2.0))
    assert dqn_row["eval_average_power_normalized_mean"] == pytest.approx(11.0)


def test_training_diagnostics_flags_non_finite_loss(tmp_path):
    path = tmp_path / "metrics.csv"
    frame = pd.DataFrame(
        {
            "loss": [1.0] * 10 + [float("nan")] * 10,
            "grad_norm": [1.0] * 20,
        }
    )
    frame.to_csv(path, index=False)
    diagnostics = training_diagnostics(path)
    assert diagnostics["diverged"] == 1
    assert diagnostics["max_grad_norm"] == pytest.approx(1.0)


def test_training_diagnostics_is_conservative_for_growing_loss(tmp_path):
    path = tmp_path / "metrics.csv"
    frame = pd.DataFrame(
        {
            "loss": [w * 0.1 for w in range(1, 41)],
            "grad_norm": [1.0] * 40,
        }
    )
    frame.to_csv(path, index=False)
    diagnostics = training_diagnostics(path)
    # Ordinary early-training growth is not a divergence.
    assert diagnostics["diverged"] == 0
    assert diagnostics["loss_growth_ratio"] > 5.0


# ---------------------------------------------------------------------------
# Controller projection
# ---------------------------------------------------------------------------

def test_continuous_command_projection():
    assert command_to_action(ACTION_DEADBAND_DEG + 0.1) == 1
    assert command_to_action(-(ACTION_DEADBAND_DEG + 0.1)) == 0
    assert command_to_action(0.0) == 2
    assert command_to_action(ACTION_DEADBAND_DEG - 0.01) == 2


def test_rule_and_pid_actions_are_discrete():
    aligned = np.array([0.0, 0.5, 0.0, 0.5, 0.5], dtype=np.float32)  # 0 deg
    misaligned = np.array([0.0, 0.5, 0.0, 0.75, 0.5], dtype=np.float32)  # +90 deg
    assert rule_action(aligned) == 2
    assert rule_action(misaligned) == 1
    assert pid_action(aligned) in (0, 1, 2)
    assert pid_action(misaligned) in (0, 1, 2)


# ---------------------------------------------------------------------------
# Checkpoint dispatch
# ---------------------------------------------------------------------------

def test_double_dqn_checkpoint_dispatches_to_double_agent(tmp_path):
    set_global_seed(11)
    agent = DoubleDQNAgent(DQNConfig(device="cpu", hidden_sizes=(16, 8)))
    path = tmp_path / "double.pth"
    agent.save(path)

    loaded = DQNAgent.from_checkpoint(path, device="cpu")
    assert isinstance(loaded, DoubleDQNAgent)
    assert loaded.config.double_dqn is True
    assert loaded.config.name == "double_dqn"
    assert loaded.config.hidden_sizes == (16, 8)


# ---------------------------------------------------------------------------
# End-to-end training loop
# ---------------------------------------------------------------------------

def test_short_training_run_writes_real_metrics(tmp_path):
    set_global_seed(21)
    env = YawRLEnvironment()
    config = DQNConfig(
        device="cpu",
        train_start_steps=0,
        batch_size=4,
        buffer_size=100,
        target_update_every=5,
        epsilon_decay_steps=2,
    )
    agent = DQNAgent(config)
    logger = MetricLogger(output_dir=tmp_path, prefix="short")
    try:
        summary = train_agent(
            env=env,
            agent=agent,
            episodes=2,
            steps_per_episode=20,
            logger=logger,
            log_interval=5,
            print_interval=0,
            seed=7,
        )
    finally:
        logger.close()

    assert summary["total_steps"] == 40
    assert summary["total_optimization_steps"] > 0
    assert math.isfinite(summary["mean_loss"])
    assert math.isfinite(summary["mean_grad_norm"])
    assert summary["total_trainable_parameters"] == 17_667

    steps = pd.read_csv(tmp_path / "short_metrics.csv")
    episodes = pd.read_csv(tmp_path / "short_episodes.csv")
    assert not steps.empty
    assert not episodes.empty
    assert steps["loss"].map(math.isfinite).all()
    assert steps["grad_norm"].map(math.isfinite).all()
    assert (steps["grad_norm"] >= 0).all()
    assert len(episodes) == 2


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_evaluate_policy_returns_real_metrics():
    env = YawRLEnvironment()
    results = evaluate_policy(
        env,
        action_fn=lambda _state: 2,  # hold
        episodes=1,
        seed_base=12_345,
        measure_inference=True,
    )
    assert len(results) == 1
    metrics = results[0]
    for key in (
        "episode_reward",
        "average_power_normalized",
        "average_power_physical",
        "mean_abs_misalignment",
        "yaw_movement_frequency",
        "control_stability",
        "time_within_10_deg",
        "inference_time_ms",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])

    assert metrics["average_power_physical"] > 0.0
    assert 0.0 <= metrics["time_within_10_deg"] <= 1.0
    # The hold policy never commands movement, so the applied yaw change is 0.
    assert metrics["yaw_movement_frequency"] == pytest.approx(0.0)
    assert metrics["inference_time_ms"] >= 0.0


def test_q_values_shape_and_greedy_consistency():
    set_global_seed(13)
    agent = DQNAgent(DQNConfig(device="cpu"))
    states = np.random.rand(4, 5).astype(np.float32)
    q_values = agent.q_values(states)
    assert q_values.shape == (4, 3)
    for row, state in zip(q_values, states):
        greedy = agent.select_action(state, greedy=True)
        assert greedy == int(np.argmax(row))
