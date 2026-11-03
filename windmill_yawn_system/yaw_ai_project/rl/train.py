"""Train the original Q-learning baseline and the PyTorch DQN agent.

The original command still works:

    python rl/train.py --q-episodes 500 --dqn-episodes 2200 --steps-per-episode 1000

New options add Double DQN, richer metrics and TensorBoard logging without
removing the original output files (``rl/model.pth`` and ``rl/logs.csv``).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import YawRLEnvironment
from rl.agents import (
    ACTION_SIZE,
    Q_BINS,
    STATE_SIZE,
    DQNConfig,
    DQNAgent,
    DoubleDQNAgent,
    QNetwork,
    ReplayBuffer,
    discretize_state,
    resolve_device,
    set_global_seed,
    train_tabular_q_learning,
)
from rl.metrics import MetricLogger
from rl.trainer import train_agent

# ---------------------------------------------------------------------------
# Original constants (kept for backward compatibility)
# ---------------------------------------------------------------------------

SEED = 42
DEFAULT_Q_EPISODES = 500
DEFAULT_DQN_EPISODES = 2200
DEFAULT_STEPS_PER_EPISODE = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1500

Q_LEARNING_RATE = 0.05
Q_GAMMA = 0.99

DQN_GAMMA = 0.99
DQN_LEARNING_RATE = 3e-4
BUFFER_SIZE = 50_000
BATCH_SIZE = 256
TARGET_UPDATE_EVERY = 250
TRAIN_START_STEPS = 2_000
GRAD_CLIP_NORM = 5.0

MODEL_PATH = os.path.join("rl", "model.pth")
LOG_PATH = os.path.join("rl", "logs.csv")
METRICS_DIR = os.path.join("results", "training")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    set_global_seed(seed)


def epsilon_by_episode(ep: int) -> float:
    if ep >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    frac = ep / float(EPSILON_DECAY_EPISODES)
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def evaluate_random_policy(env, episodes: int = 20) -> float:
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total += reward
            done = done or truncated
        rewards.append(total)
    return float(np.mean(rewards))


def train_q_learning(env, q_episodes: int, steps_per_episode: int):
    """Legacy Q-learning wrapper. Returns ``(q_table, episode_rewards)``."""

    q_table, rewards, _history = train_tabular_q_learning(
        env,
        episodes=q_episodes,
        steps_per_episode=steps_per_episode,
        learning_rate=Q_LEARNING_RATE,
        gamma=Q_GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_episodes=EPSILON_DECAY_EPISODES,
        bins=Q_BINS,
    )
    return q_table, rewards


def select_action_dqn(policy_net: QNetwork, state, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(policy_net, target_net, optimizer, replay_buffer, global_step):
    """Legacy single update helper, retained for external imports."""

    if global_step <= TRAIN_START_STEPS:
        return None
    if len(replay_buffer) < BATCH_SIZE:
        return None
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    current_q = policy_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + DQN_GAMMA * next_q * (1.0 - dones_t)
    loss = torch.nn.functional.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    return float(loss.item())


def train_dqn(env, dqn_episodes: int, steps_per_episode: int):
    """Legacy DQN wrapper. Returns ``(policy_net, episode_rewards)``.

    New code should use :class:`rl.agents.DQNAgent` directly so that the
    checkpoint, optimizer and metrics are retained.
    """

    agent, summary = _train_neural_agent(
        env=env,
        algorithm="dqn",
        episodes=dqn_episodes,
        steps_per_episode=steps_per_episode,
        seed=SEED,
        metrics_dir=METRICS_DIR,
        run_name="dqn_legacy",
        tensorboard_dir=None,
        log_interval=100,
        print_interval=50,
    )
    return agent.online, summary["episode_rewards"]


def write_logs(log_rows: List[Dict[str, object]], path: str = LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["phase", "episode", "episode_reward", "average_reward_50", "epsilon"],
        )
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# New training orchestration
# ---------------------------------------------------------------------------

def _make_config(
    algorithm: str,
    seed: int,
    device: str,
    use_replay: bool = True,
    use_target_network: bool = True,
) -> DQNConfig:
    return DQNConfig(
        gamma=DQN_GAMMA,
        learning_rate=DQN_LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_every=TARGET_UPDATE_EVERY,
        train_start_steps=TRAIN_START_STEPS,
        grad_clip_norm=GRAD_CLIP_NORM,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_EPISODES,
        double_dqn=(algorithm == "double_dqn"),
        use_replay=use_replay,
        use_target_network=use_target_network,
        device=device,
        seed=seed,
        name=algorithm,
    )


def _train_neural_agent(
    env,
    algorithm: str,
    episodes: int,
    steps_per_episode: int,
    seed: int,
    metrics_dir: str,
    run_name: str,
    tensorboard_dir: Optional[str],
    log_interval: int,
    print_interval: int,
    device: str = "auto",
) -> Tuple[DQNAgent, Dict[str, object]]:
    set_global_seed(seed)
    config = _make_config(algorithm, seed, device)
    agent = DoubleDQNAgent(config) if algorithm == "double_dqn" else DQNAgent(config)
    logger = MetricLogger(
        output_dir=metrics_dir,
        prefix=run_name,
        tensorboard_dir=tensorboard_dir,
        tensorboard_run_name=run_name,
    )
    try:
        summary = train_agent(
            env=env,
            agent=agent,
            episodes=episodes,
            steps_per_episode=steps_per_episode,
            logger=logger,
            log_interval=log_interval,
            print_interval=print_interval,
            seed=seed,
        )
    finally:
        logger.close()
    return agent, summary


def _episode_log_rows(
    phase: str,
    rewards: List[float],
    total_episodes: int,
) -> List[Dict[str, object]]:
    rows = []
    for i, reward in enumerate(rewards):
        rows.append(
            {
                "phase": phase,
                "episode": i + 1,
                "episode_reward": float(reward),
                "average_reward_50": float(np.mean(rewards[max(0, i - 49): i + 1])),
                "epsilon": float(epsilon_by_episode(i)),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-learning, DQN and/or Double DQN yaw controllers")
    parser.add_argument("--q-episodes", type=int, default=DEFAULT_Q_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_DQN_EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--eval-random-episodes", type=int, default=20)
    parser.add_argument(
        "--algorithm",
        choices=["dqn", "double_dqn", "both", "none"],
        default="dqn",
        help="Neural agent to train; 'both' trains DQN then Double DQN.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--double-model-path", default=os.path.join("rl", "model_double_dqn.pth"))
    parser.add_argument("--log-path", default=LOG_PATH)
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs("rl", exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    env = YawRLEnvironment()
    print(f"Using device: {resolve_device(args.device)}")

    random_baseline = evaluate_random_policy(env, episodes=args.eval_random_episodes)
    print(f"Random policy average reward: {random_baseline:.3f}")

    q_rewards: List[float] = []
    if args.q_episodes > 0:
        q_table, q_rewards = train_q_learning(env, args.q_episodes, args.steps_per_episode)

    neural_summaries: Dict[str, Dict[str, object]] = {}

    if args.algorithm in ("dqn", "both"):
        run_name = args.run_name or "dqn"
        dqn_agent, dqn_summary = _train_neural_agent(
            env=env,
            algorithm="dqn",
            episodes=args.dqn_episodes,
            steps_per_episode=args.steps_per_episode,
            seed=args.seed,
            metrics_dir=args.metrics_dir,
            run_name=run_name,
            tensorboard_dir=args.tensorboard_dir,
            log_interval=args.log_interval,
            print_interval=args.print_interval,
            device=args.device,
        )
        dqn_agent.save(args.model_path)
        neural_summaries["dqn"] = dqn_summary
        print(f"Saved DQN model: {args.model_path}")
        print(f"DQN Avg(Last 50): {dqn_summary['final_avg_reward_50']:.3f}")

    if args.algorithm in ("double_dqn", "both"):
        run_name = (args.run_name + "_double") if args.run_name else "double_dqn"
        double_agent, double_summary = _train_neural_agent(
            env=env,
            algorithm="double_dqn",
            episodes=args.dqn_episodes,
            steps_per_episode=args.steps_per_episode,
            seed=args.seed,
            metrics_dir=args.metrics_dir,
            run_name=run_name,
            tensorboard_dir=args.tensorboard_dir,
            log_interval=args.log_interval,
            print_interval=args.print_interval,
            device=args.device,
        )
        double_agent.save(args.double_model_path)
        neural_summaries["double_dqn"] = double_summary
        print(f"Saved Double DQN model: {args.double_model_path}")
        print(f"Double DQN Avg(Last 50): {double_summary['final_avg_reward_50']:.3f}")

    # Preserve the original combined logs.csv format.
    log_rows: List[Dict[str, object]] = []
    log_rows.extend(_episode_log_rows("q_learning", q_rewards, args.q_episodes))
    if "dqn" in neural_summaries:
        log_rows.extend(
            _episode_log_rows("dqn", neural_summaries["dqn"]["episode_rewards"], args.dqn_episodes)
        )
    if "double_dqn" in neural_summaries:
        log_rows.extend(
            _episode_log_rows(
                "double_dqn",
                neural_summaries["double_dqn"]["episode_rewards"],
                args.dqn_episodes,
            )
        )
    if log_rows:
        write_logs(log_rows, path=args.log_path)
        print(f"Saved logs: {args.log_path}")

    if "dqn" in neural_summaries:
        print(
            f"Improvement over random (DQN): "
            f"{neural_summaries['dqn']['final_avg_reward_50'] - random_baseline:.3f}"
        )
    if args.tensorboard_dir:
        print(f"TensorBoard logs: {args.tensorboard_dir}")
        print(f"Launch with: tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
