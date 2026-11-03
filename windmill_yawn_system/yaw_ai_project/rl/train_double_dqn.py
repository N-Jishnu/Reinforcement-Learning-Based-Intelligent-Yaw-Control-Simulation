"""Dedicated Double DQN training entry point."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import YawRLEnvironment
from rl.agents import set_global_seed
from rl.train import (
    DEFAULT_DQN_EPISODES,
    DEFAULT_STEPS_PER_EPISODE,
    METRICS_DIR,
    _train_neural_agent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Double DQN yaw controller")
    parser.add_argument("--episodes", type=int, default=DEFAULT_DQN_EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    parser.add_argument("--run-name", default="double_dqn")
    parser.add_argument("--model-path", default=os.path.join("rl", "model_double_dqn.pth"))
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--print-interval", type=int, default=50)
    args = parser.parse_args()

    set_global_seed(args.seed)
    env = YawRLEnvironment()
    agent, summary = _train_neural_agent(
        env=env,
        algorithm="double_dqn",
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
        metrics_dir=args.metrics_dir,
        run_name=args.run_name,
        tensorboard_dir=args.tensorboard_dir,
        log_interval=args.log_interval,
        print_interval=args.print_interval,
        device=args.device,
    )
    agent.save(args.model_path)
    print(f"Saved Double DQN model: {args.model_path}")
    print(f"Double DQN Avg(Last 50): {summary['final_avg_reward_50']:.3f}")
    print(f"Metrics: {args.metrics_dir}")
    if args.tensorboard_dir:
        print(f"TensorBoard: tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
