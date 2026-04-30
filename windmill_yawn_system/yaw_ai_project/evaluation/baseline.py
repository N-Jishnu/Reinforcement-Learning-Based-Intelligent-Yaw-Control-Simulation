import csv
import argparse
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


RUNS = 10
STEPS = 1000
DT = 1.0
RULE_THRESHOLD = 10.0
KP = 0.5
KD = 0.1
MAX_CMD_STEP = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def angle_diff(a_deg, b_deg):
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def normalize_deg(angle_deg):
    return angle_deg % 360.0


def load_rl_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = QNetwork(input_size=5, output_size=3).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_state(wd, ws, yaw, mis, wd_change):
    state = np.array(
        [
            wd / 360.0,
            (ws - 3.0) / 12.0,
            yaw / 360.0,
            (mis + 180.0) / 360.0,
            (wd_change + 180.0) / 360.0,
        ],
        dtype=np.float32,
    )
    return np.clip(state, 0.0, 1.0)


def rule_action(misalignment):
    if abs(misalignment) > RULE_THRESHOLD:
        return np.sign(misalignment) * MAX_CMD_STEP
    return 0.0


def pid_action(misalignment, wd_change):
    cmd = KP * misalignment + KD * wd_change
    return float(np.clip(cmd, -MAX_CMD_STEP, MAX_CMD_STEP))


def rl_action(model, state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = int(torch.argmax(model(s), dim=1).item())
    if action == 0:
        return -2.0
    if action == 1:
        return 2.0
    return 0.0


def run_controller(controller, wind_speed, wind_dir, rl_model):
    turbine = TurbineModel(initial_yaw=0.0)
    prev_wd = float(wind_dir[0])
    prev_yaw_change = 0.0

    p_norm = []
    p_phys = []
    yaw_changes = []

    for t in range(len(wind_speed)):
        ws = float(wind_speed[t])
        wd = float(wind_dir[t])
        yaw = turbine.get_yaw_angle()
        mis = angle_diff(wd, yaw)
        wd_change = angle_diff(wd, prev_wd)

        if controller == "rule":
            cmd = rule_action(mis)
        elif controller == "pid":
            cmd = pid_action(mis, wd_change)
        elif controller == "rl":
            state = build_state(wd, ws, yaw, mis, wd_change)
            cmd = rl_action(rl_model, state)
        else:
            raise ValueError("Unknown controller")

        target = normalize_deg(yaw + cmd)
        turbine.step(target, DT)
        yaw_new = turbine.get_yaw_angle()
        yaw_change = angle_diff(yaw_new, yaw)
        mis_new = angle_diff(wd, yaw_new)

        p_norm.append(compute_power(ws, mis_new, mode="normalized"))
        p_phys.append(compute_power(ws, mis_new, mode="physical"))
        yaw_changes.append(yaw_change)

        prev_wd = wd
        prev_yaw_change = yaw_change

    yaw_changes = np.array(yaw_changes, dtype=float)
    p_norm = np.array(p_norm, dtype=float)
    p_phys = np.array(p_phys, dtype=float)

    yaw_movement_count = int(np.sum(np.abs(yaw_changes) > 1e-6))
    oscillation_count = int(
        np.sum(
            (yaw_changes[1:] * yaw_changes[:-1] < 0)
            & (np.abs(yaw_changes[1:]) > 1e-6)
            & (np.abs(yaw_changes[:-1]) > 1e-6)
        )
    )

    return {
        "total_energy_normalized": float(np.sum(p_norm) * DT),
        "average_power_normalized": float(np.mean(p_norm)),
        "total_energy_physical": float(np.sum(p_phys) * DT),
        "average_power_physical": float(np.mean(p_phys)),
        "yaw_movement_count": yaw_movement_count,
        "oscillation_count": oscillation_count,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline comparison: Rule vs PID vs RL")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument("--data-path", type=str, default="data/era5_chennai.csv")
    parser.add_argument("--model-path", type=str, default=os.path.join("rl", "model.pth"))
    parser.add_argument("--out-path", type=str, default=os.path.join("evaluation", "results.csv"))
    return parser.parse_args()


def build_seed_schedule(args):
    if args.seed_list.strip():
        seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
        if not seeds:
            raise ValueError("seed-list was provided but no valid integers were parsed")
        return seeds
    return [args.base_seed + i for i in range(args.runs)]


def main():
    args = parse_args()
    out_path = args.out_path

    set_seed(args.base_seed)
    rl_model = load_rl_model(args.model_path)
    seed_schedule = build_seed_schedule(args)

    rows = []
    controllers = ["rule", "pid", "rl"]

    for run, run_seed in enumerate(seed_schedule, start=1):
        set_seed(run_seed)
        wind_speed, wind_dir = generate_wind_series(args.steps, 1.0, args.data_path)
        for controller in controllers:
            metrics = run_controller(controller, wind_speed, wind_dir, rl_model)
            rows.append({"run": run, "seed": run_seed, "controller": controller, **metrics})
        print(f"Completed run {run}/{len(seed_schedule)} (seed={run_seed})")

    os.makedirs("evaluation", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "seed",
                "controller",
                "total_energy_normalized",
                "average_power_normalized",
                "total_energy_physical",
                "average_power_physical",
                "yaw_movement_count",
                "oscillation_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
