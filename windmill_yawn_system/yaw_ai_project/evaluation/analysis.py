import os

import pandas as pd


def summarize(df, metric):
    out = (
        df.groupby("controller", as_index=False)[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    )
    return out


def main():
    in_path = os.path.join("evaluation", "results.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing file: {in_path}")

    df = pd.read_csv(in_path)

    metrics = [
        "total_energy_normalized",
        "average_power_normalized",
        "total_energy_physical",
        "average_power_physical",
        "yaw_movement_count",
        "oscillation_count",
    ]

    summaries = []
    for metric in metrics:
        s = summarize(df, metric)
        s["metric"] = metric
        summaries.append(s)

    summary_df = pd.concat(summaries, ignore_index=True)
    out_path = os.path.join("evaluation", "analysis_summary.csv")
    summary_df.to_csv(out_path, index=False)

    print("Statistical Analysis (mean +/- std)")
    for metric in metrics:
        sub = summary_df[summary_df["metric"] == metric]
        print(f"\n{metric}:")
        for _, row in sub.iterrows():
            val_mean = row.get(f"{metric}_mean", 0.0)
            val_std = row.get(f"{metric}_std", 0.0)
            print(f"  {row['controller']}: {val_mean:.3f} +/- {val_std:.3f}")

    # Success criteria checks
    rl = df[df["controller"] == "rl"]
    rule = df[df["controller"] == "rule"]
    pid = df[df["controller"] == "pid"]

    rl_vs_rule_norm = rl["average_power_normalized"].mean() - rule["average_power_normalized"].mean()
    rl_vs_pid_norm = rl["average_power_normalized"].mean() - pid["average_power_normalized"].mean()
    consistency = rl["average_power_normalized"].std()

    print("\nRL Comparison (normalized mode):")
    print(f"  RL - Rule average power: {rl_vs_rule_norm:.3f}")
    print(f"  RL - PID average power:  {rl_vs_pid_norm:.3f}")
    print(f"  RL run-to-run std:       {consistency:.3f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
