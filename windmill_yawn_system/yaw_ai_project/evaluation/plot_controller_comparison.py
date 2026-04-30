import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric(summary_df, mode, metric_mean, metric_std, ylabel, title, out_path):
    if mode is None:
        df = summary_df.copy()
    else:
        df = summary_df[summary_df["mode"] == mode].copy()
    controllers = df["controller"].tolist()
    means = df[metric_mean].to_numpy(dtype=float)
    stds = df[metric_std].to_numpy(dtype=float) if metric_std in df.columns else np.zeros_like(means)

    x = np.arange(len(controllers))
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, yerr=stds, capsize=6, alpha=0.85)
    plt.xticks(x, [c.upper() for c in controllers])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)

    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot controller comparison metrics")
    parser.add_argument("--summary", type=str, default=os.path.join("results", "controller_comparison_summary.csv"))
    parser.add_argument("--runs", type=str, default=os.path.join("results", "controller_comparison_runs.csv"))
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        raise FileNotFoundError(f"Summary CSV not found: {args.summary}")
    if not os.path.exists(args.runs):
        raise FileNotFoundError(f"Runs CSV not found: {args.runs}")

    os.makedirs(args.out_dir, exist_ok=True)

    summary_df = pd.read_csv(args.summary)
    runs_df = pd.read_csv(args.runs)

    for mode in ["normalized", "physical"]:
        plot_metric(
            summary_df,
            mode,
            metric_mean="average_power_mean",
            metric_std="average_power_std",
            ylabel="Average Power",
            title=f"Average Power Comparison ({mode})",
            out_path=os.path.join(args.out_dir, f"controller_avg_power_{mode}.png"),
        )
        plot_metric(
            summary_df,
            mode,
            metric_mean="total_energy_mean",
            metric_std="total_energy_std",
            ylabel="Total Energy",
            title=f"Total Energy Comparison ({mode})",
            out_path=os.path.join(args.out_dir, f"controller_total_energy_{mode}.png"),
        )

    # Cross-mode mechanics metrics (from runs, same values across modes for movement metrics)
    mech = (
        runs_df[runs_df["mode"] == "normalized"]
        .groupby("controller", as_index=False)
        .agg(
            yaw_movement_frequency_mean=("yaw_movement_frequency", "mean"),
            yaw_movement_frequency_std=("yaw_movement_frequency", "std"),
            control_stability_mean=("control_stability", "mean"),
            control_stability_std=("control_stability", "std"),
            mean_abs_misalignment_mean=("mean_abs_misalignment", "mean"),
            mean_abs_misalignment_std=("mean_abs_misalignment", "std"),
        )
    )
    mech_summary_path = os.path.join(args.out_dir, "controller_mechanics_summary.csv")
    mech.to_csv(mech_summary_path, index=False)

    plot_metric(
        mech.rename(columns={"controller": "controller", "yaw_movement_frequency_mean": "m", "yaw_movement_frequency_std": "s"}),
        mode=None,
        metric_mean="m",
        metric_std="s",
        ylabel="Yaw Movement Frequency",
        title="Yaw Movement Frequency Comparison",
        out_path=os.path.join(args.out_dir, "controller_yaw_movement_frequency.png"),
    )

    plot_metric(
        mech.rename(columns={"controller": "controller", "control_stability_mean": "m", "control_stability_std": "s"}),
        mode=None,
        metric_mean="m",
        metric_std="s",
        ylabel="Control Stability (std yaw change)",
        title="Control Stability Comparison",
        out_path=os.path.join(args.out_dir, "controller_control_stability.png"),
    )

    plot_metric(
        mech.rename(columns={"controller": "controller", "mean_abs_misalignment_mean": "m", "mean_abs_misalignment_std": "s"}),
        mode=None,
        metric_mean="m",
        metric_std="s",
        ylabel="Mean Absolute Misalignment (deg)",
        title="Misalignment Comparison",
        out_path=os.path.join(args.out_dir, "controller_misalignment.png"),
    )

    print("Saved plots:")
    print(os.path.join(args.out_dir, "controller_avg_power_normalized.png"))
    print(os.path.join(args.out_dir, "controller_avg_power_physical.png"))
    print(os.path.join(args.out_dir, "controller_total_energy_normalized.png"))
    print(os.path.join(args.out_dir, "controller_total_energy_physical.png"))
    print(os.path.join(args.out_dir, "controller_yaw_movement_frequency.png"))
    print(os.path.join(args.out_dir, "controller_control_stability.png"))
    print(os.path.join(args.out_dir, "controller_misalignment.png"))
    print(f"Saved mechanics summary: {mech_summary_path}")


if __name__ == "__main__":
    main()
