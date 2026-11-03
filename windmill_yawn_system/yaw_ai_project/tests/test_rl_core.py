"""Core neural-network, replay, agent and metric tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from env.environment import YawRLEnvironment
from rl.agents import (
    ACTION_SIZE,
    STATE_SIZE,
    DQNConfig,
    DQNAgent,
    DoubleDQNAgent,
    QNetwork,
    ReplayBuffer,
    infer_hidden_sizes,
    set_global_seed,
)
from rl.metrics import MetricLogger


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

def test_network_input_output_dimensions():
    model = QNetwork()
    output = model(torch.zeros(4, STATE_SIZE))
    assert output.shape == (4, ACTION_SIZE)
    assert model.input_size == 5
    assert model.output_size == 3
    assert model.hidden_sizes == (128, 128)


def test_network_parameter_summary_matches_pytorch():
    model = QNetwork()
    summary = model.parameter_summary()
    expected = 5 * 128 + 128 + 128 * 128 + 128 + 128 * 3 + 3
    assert summary["total_trainable_parameters"] == expected == 17_667
    assert summary["num_linear_layers"] == 3
    assert [row["out_features"] for row in summary["layers"]] == [128, 128, 3]
    assert summary["layers"][0]["activation"] == "ReLU"
    assert summary["layers"][-1]["activation"].startswith("None")


def test_infer_hidden_sizes_from_state_dict():
    model = QNetwork(hidden_sizes=(32, 16))
    assert infer_hidden_sizes(model.state_dict()) == (32, 16)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def test_replay_buffer_add_sample_and_capacity():
    buffer = ReplayBuffer(capacity=3)
    for i in range(5):
        state = np.full(STATE_SIZE, i, dtype=np.float32)
        buffer.add(state, i % 3, float(i), state + 1, bool(i == 4))
    assert len(buffer) == 3

    states, actions, rewards, next_states, dones = buffer.sample(2)
    assert states.shape == (2, STATE_SIZE)
    assert actions.shape == (2,)
    assert rewards.shape == (2,)
    assert next_states.shape == (2, STATE_SIZE)
    assert dones.shape == (2,)


def test_replay_buffer_rejects_oversized_sample():
    buffer = ReplayBuffer(capacity=10)
    buffer.add(np.zeros(STATE_SIZE), 0, 0.0, np.zeros(STATE_SIZE), False)
    with pytest.raises(ValueError):
        buffer.sample(2)


# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------

def test_dqn_greedy_action_selection_is_deterministic():
    set_global_seed(1)
    agent = DQNAgent(DQNConfig(device="cpu", seed=1))
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    actions = {agent.select_action(state, epsilon=1.0, greedy=True) for _ in range(20)}
    assert len(actions) == 1
    assert 0 <= actions.pop() < ACTION_SIZE


def test_dqn_exploration_returns_valid_actions():
    set_global_seed(2)
    agent = DQNAgent(DQNConfig(device="cpu"))
    state = np.zeros(STATE_SIZE, dtype=np.float32)
    actions = {agent.select_action(state, epsilon=1.0) for _ in range(100)}
    assert actions.issubset({0, 1, 2})


def test_dqn_training_update_reports_real_metrics():
    set_global_seed(3)
    config = DQNConfig(
        device="cpu",
        train_start_steps=0,
        batch_size=4,
        buffer_size=100,
        target_update_every=10,
    )
    agent = DQNAgent(config)
    for i in range(10):
        state = np.random.rand(STATE_SIZE).astype(np.float32)
        agent.remember(state, i % 3, float(i) / 10.0, np.random.rand(STATE_SIZE), i == 9)
    agent.global_step = 1

    metrics = agent.optimize()
    assert metrics is not None
    assert set(metrics) == {"loss", "grad_norm", "avg_q", "max_q"}
    assert math.isfinite(metrics["loss"]) and metrics["loss"] >= 0.0
    assert math.isfinite(metrics["grad_norm"]) and metrics["grad_norm"] > 0.0
    assert math.isfinite(metrics["avg_q"]) and math.isfinite(metrics["max_q"])
    assert agent.optimization_steps == 1


# ---------------------------------------------------------------------------
# Double DQN
# ---------------------------------------------------------------------------

def test_double_dqn_target_uses_online_argmax_and_target_evaluation():
    set_global_seed(4)
    config = DQNConfig(device="cpu", double_dqn=True, gamma=0.9)
    agent = DoubleDQNAgent(config)

    # Force different online/target weights so the formula is exercised.
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.add_(0.01)
        for parameter in agent.target.parameters():
            parameter.sub_(0.01)

    next_states = torch.rand(6, STATE_SIZE)
    rewards = torch.rand(6, 1)
    dones = torch.tensor([[0.0], [1.0], [0.0], [0.0], [1.0], [0.0]])

    with torch.no_grad():
        online_actions = agent.online(next_states).argmax(dim=1, keepdim=True)
        expected = rewards + config.gamma * agent.target(next_states).gather(1, online_actions) * (1.0 - dones)

    actual = agent.compute_target(rewards, next_states, dones)
    assert torch.allclose(actual, expected, atol=1e-6)
    # Terminal transitions must not bootstrap.
    terminal_rows = dones.squeeze(1) == 1
    assert torch.allclose(actual[terminal_rows], rewards[terminal_rows], atol=1e-6)


def test_standard_dqn_target_uses_max_of_target_network():
    set_global_seed(5)
    config = DQNConfig(device="cpu", double_dqn=False, gamma=0.8)
    agent = DQNAgent(config)
    next_states = torch.rand(4, STATE_SIZE)
    rewards = torch.zeros(4, 1)
    dones = torch.zeros(4, 1)

    with torch.no_grad():
        expected = config.gamma * agent.target(next_states).max(dim=1, keepdim=True)[0]
    actual = agent.compute_target(rewards, next_states, dones)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_target_network_synchronization():
    set_global_seed(6)
    agent = DQNAgent(DQNConfig(device="cpu"))
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.add_(0.5)
    assert not torch.allclose(
        agent.online.net[0].weight, agent.target.net[0].weight
    )
    agent.update_target_network()
    for online_parameter, target_parameter in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.allclose(online_parameter, target_parameter)


# ---------------------------------------------------------------------------
# Checkpoints and reproducibility
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_load(tmp_path):
    set_global_seed(7)
    agent = DQNAgent(DQNConfig(device="cpu", hidden_sizes=(32, 16)))
    path = tmp_path / "checkpoint.pth"
    agent.save(path)
    assert path.exists()

    loaded = DQNAgent.from_checkpoint(path, device="cpu")
    assert loaded.config.hidden_sizes == (32, 16)
    state = np.random.rand(STATE_SIZE).astype(np.float32)
    with torch.no_grad():
        expected = agent.online(torch.as_tensor(state).unsqueeze(0))
        actual = loaded.online(torch.as_tensor(state).unsqueeze(0))
    assert torch.allclose(expected, actual)


def test_reproducibility_of_network_initialisation():
    set_global_seed(123)
    first = QNetwork()(torch.rand(2, STATE_SIZE))
    set_global_seed(123)
    second = QNetwork()(torch.rand(2, STATE_SIZE))
    assert torch.allclose(first, second)


def test_reproducibility_of_action_sequence():
    set_global_seed(99)
    agent = DQNAgent(DQNConfig(device="cpu"))
    state = np.full(STATE_SIZE, 0.3, dtype=np.float32)
    first = [agent.select_action(state, epsilon=0.5) for _ in range(10)]

    set_global_seed(99)
    agent = DQNAgent(DQNConfig(device="cpu"))
    second = [agent.select_action(state, epsilon=0.5) for _ in range(10)]
    assert first == second


# ---------------------------------------------------------------------------
# Environment reproducibility
# ---------------------------------------------------------------------------

def test_environment_reset_is_seeded():
    env = YawRLEnvironment()
    env.reset(seed=555)
    first_speed = env.wind_speed_series.copy()
    first_direction = env.wind_direction_series.copy()
    env.reset(seed=555)
    assert np.allclose(first_speed, env.wind_speed_series)
    assert np.allclose(first_direction, env.wind_direction_series)


def test_environment_info_contains_wind_fields():
    env = YawRLEnvironment()
    env.reset(seed=1)
    _state, _reward, _terminated, _truncated, info = env.step(2)
    assert "wind_speed" in info
    assert "wind_direction" in info


# ---------------------------------------------------------------------------
# Metric logging
# ---------------------------------------------------------------------------

def test_metric_logging_writes_csv_files(tmp_path):
    logger = MetricLogger(output_dir=tmp_path, prefix="unit")
    logger.log_step(
        episode=1,
        global_step=10,
        loss=0.5,
        grad_norm=1.2,
        avg_q=0.1,
        max_q=0.3,
        epsilon=0.9,
        learning_rate=3e-4,
    )
    logger.log_episode(
        episode=1,
        episode_reward=12.0,
        moving_avg_50=12.0,
        epsilon=0.9,
        episode_length=100,
        episode_time_s=0.2,
        total_time_s=0.2,
        optimization_steps=5,
        mean_loss=0.5,
        mean_grad_norm=1.2,
        mean_q=0.1,
        max_q=0.3,
        learning_rate=3e-4,
    )
    logger.log_summary({"algorithm": "dqn", "seed": 0, "wall_time_s": 1.0})
    logger.close()

    steps = pd.read_csv(tmp_path / "unit_metrics.csv")
    episodes = pd.read_csv(tmp_path / "unit_episodes.csv")
    summary = pd.read_csv(tmp_path / "training_summary.csv")
    assert len(steps) == 1 and steps.loc[0, "loss"] == pytest.approx(0.5)
    assert len(episodes) == 1 and episodes.loc[0, "episode_reward"] == pytest.approx(12.0)
    assert len(summary) == 1 and summary.loc[0, "algorithm"] == "dqn"
