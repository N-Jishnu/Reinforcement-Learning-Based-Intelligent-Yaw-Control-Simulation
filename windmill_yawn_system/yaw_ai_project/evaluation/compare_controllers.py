import argparse
import csv
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.train import QNetwork
from simulation.power_model import compute_power
from simulation.turbine_model import TurbineModel
from simulation.wind_model import generate_wind_series


DT = 1.0
STEPS = 1000
RUNS = 10

RULE_THRESHOLD_DEG = 10.0
RULE_STEP_DEG = 2.0

PID_KP = 0.5
PID_KD = 0.1
PID_MAX_DELTA = 2.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def angle_diff(a_deg, b_deg):
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def normalize_360(angle_deg):
    return angle_deg % 360.0


def build_state(wind_direction, wind_speed, yaw_angle, misalignment, wind_direction_change):
    state = np.array(
        [
            wind_direction / 360.0,
            (wind_speed - 3.0) / 12.0,
            yaw_angle / 360.0,
            (misalignment + 180.0) / 360.0,
            (wind_direction_change + 180.0) / 360.0,
        ],
        dtype=np.float32,
    )
    return np.clip(state, 0.0, 1.0)


def load_rl_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = QNetwork(input_size=5, output_size=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def rule_controller(misalignment):
    if abs(misalignment) > RULE_THRESHOLD_DEG:
        return np.sign(misalignment) * RULE_STEP_DEG
    return 0.0


def pid_controller(misalignment, wind_dir_change):
    delta = PID_KP * misalignment + PID_KD * wind_dir_change
    return float(np.clip(delta, -PID_MAX_DELTA, PID_MAX_DELTA))


def rl_controller(model, state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = int(torch.argmax(model(s), dim=1).item())
    if action == 0:
        return -2.0
    if action == 1:
        return 2.0
    return 0.0


def run_single(controller_name, wind_speed, wind_direction, rl_model):
    turbine = TurbineModel(initial_yaw=0.0)

    yaw_history = []
    misalign_history = []
    yaw_change_history = []
    power_norm_history = []
    power_phys_history = []

    prev_wind_dir = float(wind_direction[0])
    prev_yaw = turbine.get_yaw_angle()

    for t in range(len(wind_speed)):
        wd = float(wind_direction[t])
        ws = float(wind_speed[t])
        yaw = turbine.get_yaw_angle()
        misalignment = angle_diff(wd, yaw)
        wd_change = angle_diff(wd, prev_wind_dir)

        if controller_name == "rule":
            yaw_delta = rule_controller(misalignment)
        elif controller_name == "pid":
            yaw_delta = pid_controller(misalignment, wd_change)
        elif controller_name == "rl":
            state = build_state(wd, ws, yaw, misalignment, wd_change)
            yaw_delta = rl_controller(rl_model, state)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        target_yaw = normalize_360(yaw + yaw_delta)
        turbine.step(target_yaw, DT)
        new_yaw = turbine.get_yaw_angle()
        yaw_change = angle_diff(new_yaw, prev_yaw)

        new_misalignment = angle_diff(wd, new_yaw)
        power_norm = compute_power(ws, new_misalignment, mode="normalized")
        power_phys = compute_power(ws, new_misalignment, mode="physical")

        yaw_history.append(new_yaw)
        misalign_history.append(new_misalignment)
        yaw_change_history.append(yaw_change)
        power_norm_history.append(power_norm)
        power_phys_history.append(power_phys)

        prev_wind_dir = wd
        prev_yaw = new_yaw

    return {
        "yaw": np.array(yaw_history, dtype=float),
        "misalignment": np.array(misalign_history, dtype=float),
        "yaw_change": np.array(yaw_change_history, dtype=float),
        "power_normalized": np.array(power_norm_history, dtype=float),
        "power_physical": np.array(power_phys_history, dtype=float),
    }


def compute_metrics(run_output, mode):
    power = run_output["power_normalized"] if mode == "normalized" else run_output["power_physical"]
    yaw_change = run_output["yaw_change"]
    misalignment = run_output["misalignment"]

    total_energy = float(np.sum(power) * DT)
    average_power = float(np.mean(power))
    yaw_movement_frequency = float(np.mean(np.abs(yaw_change) > 1e-6))
    control_stability = float(np.std(yaw_change))
    mean_abs_misalignment = float(np.mean(np.abs(misalignment)))

    return {
        "total_energy": total_energy,
        "average_power": average_power,
        "yaw_movement_frequency": yaw_movement_frequency,
        "control_stability": control_stability,
        "mean_abs_misalignment": mean_abs_misalignment,
    }


def summarize(rows, controllers, modes):
    summary = []
    for mode in modes:
        for ctrl in controllers:
            subset = [r for r in rows if r["mode"] == mode and r["controller"] == ctrl]
            summary.append(
                {
                    "mode": mode,
                    "controller": ctrl,
                    "total_energy_mean": float(np.mean([r["total_energy"] for r in subset])),
                    "total_energy_std": float(np.std([r["total_energy"] for r in subset])),
                    "average_power_mean": float(np.mean([r["average_power"] for r in subset])),
                    "average_power_std": float(np.std([r["average_power"] for r in subset])),
                    "yaw_movement_frequency_mean": float(np.mean([r["yaw_movement_frequency"] for r in subset])),
                    "control_stability_mean": float(np.mean([r["control_stability"] for r in subset])),
                    "mean_abs_misalignment_mean": float(np.mean([r["mean_abs_misalignment"] for r in subset])),
                }
            )
    return summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Compare Rule, PID, and RL controllers with both power modes")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, default="data/era5_chennai.csv")
    parser.add_argument("--model-path", type=str, default=os.path.join("rl", "model.pth"))
    args = parser.parse_args()

    set_seed(args.seed)
    rl_model = load_rl_model(args.model_path)

    controllers = ["rule", "pid", "rl"]
    modes = ["normalized", "physical"]
    all_rows = []

    for run_id in range(1, args.runs + 1):
        wind_speed, wind_direction = generate_wind_series(
            duration=args.steps,
            dt=1.0,
            data_path=args.data_path,
        )
        for controller in controllers:
            out = run_single(controller, wind_speed, wind_direction, rl_model)
            for mode in modes:
                metrics = compute_metrics(out, mode)
                row = {
                    "run": run_id,
                    "controller": controller,
                    "mode": mode,
                    **metrics,
                }
                all_rows.append(row)

        print(f"Completed run {run_id}/{args.runs}")

    summary = summarize(all_rows, controllers, modes)

    raw_path = os.path.join("results", "controller_comparison_runs.csv")
    summary_path = os.path.join("results", "controller_comparison_summary.csv")

    write_csv(
        raw_path,
        all_rows,
        [
            "run",
            "controller",
            "mode",
            "total_energy",
            "average_power",
            "yaw_movement_frequency",
            "control_stability",
            "mean_abs_misalignment",
        ],
    )

    write_csv(
        summary_path,
        summary,
        [
            "mode",
            "controller",
            "total_energy_mean",
            "total_energy_std",
            "average_power_mean",
            "average_power_std",
            "yaw_movement_frequency_mean",
            "control_stability_mean",
            "mean_abs_misalignment_mean",
        ],
    )

    print(f"Saved run metrics: {raw_path}")
    print(f"Saved summary: {summary_path}")

    for row in summary:
        print(
            f"[{row['mode']}] {row['controller']}: "
            f"AvgPower={row['average_power_mean']:.3f}, "
            f"Energy={row['total_energy_mean']:.3f}, "
            f"YawFreq={row['yaw_movement_frequency_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
