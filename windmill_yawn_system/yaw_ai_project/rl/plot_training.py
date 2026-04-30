import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot RL training convergence from logs.csv")
    parser.add_argument("--log-path", type=str, default=os.path.join("rl", "logs.csv"))
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.log_path)

    q_df = df[df["phase"] == "q_learning"].copy()
    dqn_df = df[df["phase"] == "dqn"].copy()

    # Plot 1: Episode reward curves
    plt.figure(figsize=(12, 6))
    if not q_df.empty:
        plt.plot(q_df["episode"], q_df["episode_reward"], alpha=0.25, label="Q-learning episode reward")
        plt.plot(q_df["episode"], q_df["average_reward_50"], linewidth=2, label="Q-learning avg(50)")
    if not dqn_df.empty:
        plt.plot(dqn_df["episode"], dqn_df["episode_reward"], alpha=0.25, label="DQN episode reward")
        plt.plot(dqn_df["episode"], dqn_df["average_reward_50"], linewidth=2, label="DQN avg(50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "training_reward_convergence.png")
    plt.savefig(out1, dpi=150)
    plt.close()

    # Plot 2: Epsilon schedules
    plt.figure(figsize=(12, 5))
    if not q_df.empty:
        plt.plot(q_df["episode"], q_df["epsilon"], label="Q-learning epsilon")
    if not dqn_df.empty:
        plt.plot(dqn_df["episode"], dqn_df["epsilon"], label="DQN epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Decay")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "training_epsilon_decay.png")
    plt.savefig(out2, dpi=150)
    plt.close()

    print(f"Saved: {out1}")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
