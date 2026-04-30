import os

import matplotlib.pyplot as plt
import pandas as pd


PLOTS_DIR = os.path.join("visualization", "plots")
RL_LOGS = os.path.join("rl", "logs.csv")
EVAL_RESULTS = os.path.join("evaluation", "results.csv")


def ensure_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_reward_vs_episodes():
    if not os.path.exists(RL_LOGS):
        print(f"Skipped reward plot: missing {RL_LOGS}")
        return

    df = pd.read_csv(RL_LOGS)
    dqn = df[df["phase"] == "dqn"].copy()

    if dqn.empty:
        print("Skipped reward plot: no DQN rows found")
        return

    plt.figure(figsize=(11, 5))
    plt.plot(dqn["episode"], dqn["episode_reward"], alpha=0.25, label="Episode reward (normalized)")
    plt.plot(dqn["episode"], dqn["average_reward_50"], linewidth=2, label="Average reward (50-episode)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (normalized)")
    plt.title("Normalized Mode: Reward vs Episodes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "normalized_reward_vs_episodes.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_energy_comparison_physical(eval_df):
    grp = eval_df.groupby("controller", as_index=False).agg(
        mean=("total_energy_physical", "mean"),
        std=("total_energy_physical", "std"),
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grp["controller"].str.upper(), grp["mean"], yerr=grp["std"], capsize=6, alpha=0.85)
    plt.xlabel("Controller")
    plt.ylabel("Total Energy (J)")
    plt.title("Physical Mode: Energy Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "physical_energy_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_power_comparison_physical(eval_df):
    grp = eval_df.groupby("controller", as_index=False).agg(
        mean=("average_power_physical", "mean"),
        std=("average_power_physical", "std"),
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grp["controller"].str.upper(), grp["mean"], yerr=grp["std"], capsize=6, alpha=0.85)
    plt.xlabel("Controller")
    plt.ylabel("Average Power (W)")
    plt.title("Physical Mode: Power Output Comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "physical_power_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    ensure_dirs()
    plot_reward_vs_episodes()

    if not os.path.exists(EVAL_RESULTS):
        print(f"Skipped physical plots: missing {EVAL_RESULTS}")
        return

    eval_df = pd.read_csv(EVAL_RESULTS)
    plot_energy_comparison_physical(eval_df)
    plot_power_comparison_physical(eval_df)


if __name__ == "__main__":
    main()
