import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.train import QNetwork
from simulation.turbine_model import TurbineModel
from simulation.wind_model import generate_wind_series


def angle_diff(a_deg, b_deg):
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def normalize_deg(a_deg):
    return a_deg % 360.0


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


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = QNetwork(5, 3).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def policy_action(controller, model, device, wd, ws, yaw, mis, wd_change):
    if controller == "Rule":
        if abs(mis) > 10.0:
            return np.sign(mis) * 2.0
        return 0.0
    if controller == "PID":
        cmd = 0.5 * mis + 0.1 * wd_change
        return float(np.clip(cmd, -2.0, 2.0))
    state = build_state(wd, ws, yaw, mis, wd_change)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a = int(torch.argmax(model(s), dim=1).item())
    if a == 0:
        return -2.0
    if a == 1:
        return 2.0
    return 0.0


def simulate_trace(controller, model, device, data_path, steps=300):
    wind_speed, wind_dir = generate_wind_series(steps, 1.0, data_path)
    turbine = TurbineModel(initial_yaw=0.0)

    yaw = []
    mis = []
    p_norm = []
    prev_wd = float(wind_dir[0])

    for t in range(steps):
        wd = float(wind_dir[t])
        ws = float(wind_speed[t])
        y = turbine.get_yaw_angle()
        m = angle_diff(wd, y)
        wd_change = angle_diff(wd, prev_wd)
        cmd = policy_action(controller, model, device, wd, ws, y, m, wd_change)

        target = normalize_deg(y + cmd)
        turbine.step(target, 1.0)
        y_new = turbine.get_yaw_angle()
        m_new = angle_diff(wd, y_new)
        power_norm = 0.5 * (ws ** 3) * max(np.cos(np.radians(m_new)) ** 3, 0.0)

        yaw.append(y_new)
        mis.append(m_new)
        p_norm.append(power_norm)
        prev_wd = wd

    return np.arange(steps), wind_dir, np.array(yaw), np.array(p_norm), np.array(mis)


def main():
    st.set_page_config(page_title="Yaw Control Dashboard", layout="wide")
    st.title("Wind Turbine Yaw Control Dashboard")
    st.caption("Normalized and physical results are displayed separately.")

    model_path_default = os.path.join(PROJECT_ROOT, "rl", "model.pth")
    data_path_default = os.path.join(PROJECT_ROOT, "data", "era5_chennai.csv")
    data_path = st.text_input("Wind dataset path", value=data_path_default)

    if not os.path.exists(data_path):
        st.error(f"Dataset not found: {data_path}")
        st.info("Set 'Wind dataset path' to a valid CSV containing wind_speed and wind_direction columns.")
        return

    model_path = st.text_input("RL model path", value=model_path_default)
    model_available = os.path.exists(model_path)

    model = None
    device = torch.device("cpu")
    if model_available:
        model, device = load_model(model_path)
        st.write(f"Inference device: `{device}`")
    else:
        st.warning(
            f"Model not found at `{model_path}`. "
            "Rule/PID dashboards still work. RL trace will be disabled until model is available."
        )

    col1, col2 = st.columns(2)
    with col1:
        controller_options = ["Rule", "PID"] if not model_available else ["RL", "Rule", "PID"]
        default_index = 0 if not model_available else 0
        controller = st.selectbox("Controller trace", controller_options, index=default_index)
    with col2:
        steps = st.slider("Trace length (steps)", min_value=100, max_value=1000, value=300, step=50)

    t, wind_dir, yaw, p_norm, mis = simulate_trace(controller, model, device, data_path=data_path, steps=steps)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, wind_dir, label="Wind direction (deg)", alpha=0.6)
    ax1.plot(t, yaw, label=f"{controller} yaw (deg)")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title("Wind vs Yaw")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t, p_norm, color="tab:green")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Normalized power")
    ax2.set_title("Power Output (Normalized Mode)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

    results_path_default = os.path.join(PROJECT_ROOT, "evaluation", "results.csv")
    results_path = st.text_input("Evaluation results path", value=results_path_default)
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        summary = (
            df.groupby("controller", as_index=False)
            .agg(
                avg_power_norm=("average_power_normalized", "mean"),
                avg_power_phys=("average_power_physical", "mean"),
                total_energy_phys=("total_energy_physical", "mean"),
            )
        )
        st.subheader("Controller Comparison")
        st.dataframe(summary)
    else:
        st.warning(f"Missing evaluation file: {results_path}")
        st.info("Run `python evaluation/baseline.py` to generate evaluation/results.csv, or set a valid CSV path above.")


if __name__ == "__main__":
    main()
