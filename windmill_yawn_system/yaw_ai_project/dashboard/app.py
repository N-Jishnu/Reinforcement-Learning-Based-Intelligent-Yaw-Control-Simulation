"""Streamlit dashboard for yaw control, neural-network training and evaluation.

The original controller-trace view is preserved.  Two new areas are added:

* ``NEURAL NETWORK TRAINING ANALYSIS`` - architecture, parameter summary,
  reward/loss/epsilon/Q-value/gradient curves, DQN vs Double DQN, ablation and
  multi-seed results.
* ``CONTROLLER PERFORMANCE COMPARISON`` - Random, Q-learning, DQN, Double DQN,
  Rule and PID in both normalized and physical power modes.

Training metrics and evaluation metrics are always shown in separate sections
so that a training reward is never confused with physical power.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.agents import QNetwork, infer_hidden_sizes
from rl.temporal_agents import LSTMDQNAgent, SequenceBuilder
from simulation.power_model import compute_power
from simulation.turbine_model import TurbineModel
from simulation.wind_model import generate_wind_series


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Load a DQN checkpoint and infer its real hidden-layer widths."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", {})
    hidden_sizes = infer_hidden_sizes(state_dict) or (128, 128)
    model = QNetwork(5, 3, hidden_sizes=hidden_sizes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device, checkpoint


def policy_action(controller, models, device, wd, ws, yaw, mis, wd_change, rng):
    if controller == "Rule":
        if abs(mis) > 10.0:
            return float(np.sign(mis) * 2.0)
        return 0.0
    if controller == "PID":
        command = 0.5 * mis + 0.1 * wd_change
        return float(np.clip(command, -2.0, 2.0))
    if controller == "Random":
        return float(rng.choice([-2.0, 0.0, 2.0]))

    model = models.get(controller)
    state = build_state(wd, ws, yaw, mis, wd_change)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = int(torch.argmax(model(s), dim=1).item())
    if action == 0:
        return -2.0
    if action == 1:
        return 2.0
    return 0.0


def simulate_trace(controller, models, temporal_models, device, data_path, steps=300, seed=42):
    wind_speed, wind_dir = generate_wind_series(steps, 1.0, data_path, seed=seed)
    turbine = TurbineModel(initial_yaw=0.0)
    rng = np.random.default_rng(seed)

    yaw = []
    mis = []
    p_norm = []
    p_phys = []
    prev_wd = float(wind_dir[0])
    temporal_agent = temporal_models.get(controller)
    sequence_builder = None

    for t in range(steps):
        wd = float(wind_dir[t])
        ws = float(wind_speed[t])
        y = turbine.get_yaw_angle()
        m = angle_diff(wd, y)
        wd_change = angle_diff(wd, prev_wd)
        if temporal_agent is not None:
            state = build_state(wd, ws, y, m, wd_change)
            if sequence_builder is None:
                sequence_builder = SequenceBuilder(
                    temporal_agent.config.sequence_length,
                    temporal_agent.config.state_size,
                )
                sequence = sequence_builder.reset(state)
            else:
                sequence = sequence_builder.append(state)
            action = temporal_agent.select_action(sequence, greedy=True)
            command = (-2.0, 2.0, 0.0)[action]
        else:
            command = policy_action(controller, models, device, wd, ws, y, m, wd_change, rng)

        target = normalize_deg(y + command)
        turbine.step(target, 1.0)
        y_new = turbine.get_yaw_angle()
        m_new = angle_diff(wd, y_new)
        p_norm.append(compute_power(ws, m_new, mode="normalized"))
        p_phys.append(compute_power(ws, m_new, mode="physical"))

        yaw.append(y_new)
        mis.append(m_new)
        prev_wd = wd

    return (
        np.arange(steps),
        np.asarray(wind_dir),
        np.asarray(yaw),
        np.asarray(p_norm),
        np.asarray(p_phys),
        np.asarray(mis),
    )


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    return frame if not frame.empty else None


def rolling(series, window=20):
    return pd.Series(series).rolling(window=window, min_periods=1).mean()


def new_figure(width=10, height=4):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, alpha=0.3)
    return fig, ax


def apply_dashboard_theme(dark_mode: bool) -> None:
    """Apply the research-dashboard visual system without changing Streamlit data flow."""

    colors = {
        "canvas": "#0B1620" if dark_mode else "#F5F8FA",
        "surface": "#142431" if dark_mode else "#FFFFFF",
        "surface_soft": "#1C3242" if dark_mode else "#EEF4F7",
        "ink": "#E6F1F7" if dark_mode else "#102A43",
        "muted": "#A8BBC8" if dark_mode else "#486581",
        "border": "#345064" if dark_mode else "#D9E2EC",
        "teal": "#38B2AC" if dark_mode else "#007C7A",
        "blue": "#60A5FA" if dark_mode else "#2563EB",
    }
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {colors['canvas']}; color: {colors['ink']}; }}
        .block-container {{ max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem; }}
        [data-testid="stHeader"] {{ background: {colors['canvas']}; }}
        .dashboard-hero {{
            background: {colors['surface']}; border: 1px solid {colors['border']};
            border-left: 5px solid {colors['teal']}; border-radius: 12px;
            padding: 1.35rem 1.5rem; margin: 0 0 1rem 0;
            box-shadow: 0 1px 3px rgba(16, 42, 67, .12);
        }}
        .dashboard-hero h1 {{ color: {colors['ink']}; font-size: 2rem; margin: 0; letter-spacing: -.02em; }}
        .dashboard-hero p {{ color: {colors['muted']}; margin: .3rem 0 0; }}
        h2, h3, h4, [data-testid="stMarkdownContainer"] {{ color: {colors['ink']}; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {colors['surface']}; border-color: {colors['border']};
            border-radius: 12px; padding: .55rem 1rem 1rem;
            box-shadow: 0 1px 3px rgba(16, 42, 67, .10);
        }}
        [data-testid="stExpander"] {{ background: {colors['surface']}; border-color: {colors['border']}; border-radius: 10px; }}
        [data-testid="stMetric"] {{ background: {colors['surface_soft']}; border: 1px solid {colors['border']}; border-radius: 10px; padding: .65rem .85rem; }}
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{ color: {colors['ink']}; }}
        [data-baseweb="tab-list"] {{ gap: .4rem; border-bottom-color: {colors['border']}; }}
        button[data-baseweb="tab"] {{ color: {colors['muted']}; border-radius: 7px 7px 0 0; }}
        button[data-baseweb="tab"][aria-selected="true"] {{ color: {colors['teal']}; }}
        [data-testid="stDataFrame"] {{ border: 1px solid {colors['border']}; border-radius: 8px; overflow: hidden; }}
        [data-testid="stCaptionContainer"], .stCaption {{ color: {colors['muted']} !important; }}
        .stButton > button, [data-testid="stDownloadButton"] > button {{ border-color: {colors['teal']}; color: {colors['teal']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    plt.rcParams.update(
        {
            "figure.facecolor": colors["surface"],
            "axes.facecolor": colors["surface"],
            "axes.edgecolor": colors["border"],
            "axes.labelcolor": colors["ink"],
            "axes.titlecolor": colors["ink"],
            "xtick.color": colors["ink"],
            "ytick.color": colors["ink"],
            "text.color": colors["ink"],
            "legend.facecolor": colors["surface"],
            "legend.edgecolor": colors["border"],
            "grid.color": colors["border"],
        }
    )


# ---------------------------------------------------------------------------
# Trace section (original functionality)
# ---------------------------------------------------------------------------

def render_trace_section(models, temporal_models, device, data_path):
    st.header("Controller trace")
    st.caption(
        "A single wind trace is replayed through Rule, PID, Random, DQN or Double DQN. "
        "The trace uses the requested controller only; comparison statistics are below."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        options = ["Rule", "PID", "Random"]
        options.extend([name for name in ("DQN", "Double DQN") if name in models])
        options.extend([name for name in ("LSTM-DQN", "Attention-LSTM-DQN") if name in temporal_models])
        controller = st.selectbox("Controller", options)
    with col2:
        steps = st.slider("Trace length (steps)", min_value=100, max_value=1000, value=300, step=50)
    with col3:
        seed = st.number_input("Trace seed", min_value=0, max_value=1_000_000, value=42, step=1)

    t, wind_dir, yaw, p_norm, p_phys, mis = simulate_trace(
        controller,
        models,
        temporal_models,
        device,
        data_path=data_path,
        steps=steps,
        seed=int(seed),
    )

    metric_a, metric_b = st.columns(2)
    metric_a.metric("Mean absolute misalignment", f"{np.mean(np.abs(mis)):.2f}°")
    metric_b.metric("Mean power proxy", f"{np.mean(p_norm):.2f}", help="Normalized power is a cubic wind-speed and yaw-alignment proxy, not a percentage.")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(t, wind_dir, label="Wind direction (deg)", alpha=0.6)
    axes[0].plot(t, yaw, label=f"{controller} yaw (deg)")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Wind vs Yaw")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, p_norm, color="tab:green", label="Normalized power")
    axes[1].set_ylabel("Normalized power")
    axes[1].set_title("Power Output (normalized mode)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, p_phys / 1000.0, color="tab:orange", label="Physical power (kW)")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Power (kW)")
    axes[2].set_title("Power Output (physical mode)")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Neural-network training analysis
# ---------------------------------------------------------------------------

def render_training_section():
    st.header("Neural network training analysis")
    st.caption(
        "All curves below are generated from real training logs. Step-level metrics "
        "(loss, Q-values, gradient norm) are logged per optimisation step; reward and "
        "epsilon are episode-level."
    )

    metrics_dir = Path(
        st.text_input(
            "Training metrics directory",
            value=str(PROJECT_ROOT / "results" / "training"),
        )
    )
    architecture_model = Path(
        st.text_input(
            "Architecture checkpoint",
            value=str(PROJECT_ROOT / "rl" / "model.pth"),
        )
    )

    # Architecture ---------------------------------------------------------
    st.subheader("1. Network architecture and parameters")
    if architecture_model.exists():
        try:
            model, device, checkpoint = load_model(architecture_model)
            st.text(model.describe())
            st.dataframe(pd.DataFrame(model.layer_summary()), width="stretch")
            st.write(
                {
                    "algorithm": checkpoint.get("algorithm", "unknown"),
                    "device": str(device),
                    "optimizer": "Adam",
                    "loss": "MSE",
                }
            )
        except Exception as exc:
            st.warning(f"Could not load architecture checkpoint: {exc}")
    else:
        st.info(f"Checkpoint not found: {architecture_model}")

    architecture_image = PROJECT_ROOT / "results" / "plots" / "network_architecture.png"
    if architecture_image.exists():
        st.image(str(architecture_image), caption="Actual Q-network topology")

    # Training curves ------------------------------------------------------
    st.subheader("2. Training curves")
    dqn_episodes = read_csv(metrics_dir / "dqn_episodes.csv")
    dqn_steps = read_csv(metrics_dir / "dqn_metrics.csv")
    double_episodes = read_csv(metrics_dir / "double_dqn_episodes.csv")
    double_steps = read_csv(metrics_dir / "double_dqn_metrics.csv")

    if dqn_episodes is None and double_episodes is None:
        st.info(
            f"No training CSV files found in {metrics_dir}. Run "
            "`python rl/train.py --algorithm both ...` first."
        )
        return

    summary_columns = st.columns(3)
    available = [label for frame, label in ((dqn_episodes, "DQN"), (double_episodes, "Double DQN")) if frame is not None]
    summary_columns[0].metric("Training logs available", str(len(available)))
    for slot, (frame, label) in zip(summary_columns[1:], ((dqn_episodes, "DQN"), (double_episodes, "Double DQN"))):
        if frame is not None and "moving_avg_50" in frame:
            slot.metric(f"{label} final training reward", f"{frame['moving_avg_50'].iloc[-1]:.2f}")

    tabs = st.tabs(
        [
            "Reward",
            "TD loss",
            "Exploration & Q-values",
            "Gradient norm",
            "DQN vs Double DQN",
            "Ablation study",
            "Multi-seed evaluation",
        ]
    )

    with tabs[0]:
        fig, ax = new_figure()
        for frame, label in ((dqn_episodes, "DQN"), (double_episodes, "Double DQN")):
            if frame is None:
                continue
            ax.plot(frame["episode"], frame["episode_reward"], alpha=0.2, label=f"{label} reward")
            ax.plot(frame["episode"], frame["moving_avg_50"], linewidth=2, label=f"{label} avg(50)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode reward")
        ax.set_title("Episode reward and moving average")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[1]:
        fig, ax = new_figure()
        for frame, label in ((dqn_steps, "DQN"), (double_steps, "Double DQN")):
            if frame is None:
                continue
            ax.plot(frame["global_step"], rolling(frame["loss"]), linewidth=2, label=f"{label} loss (rolling 20)")
        ax.set_xlabel("Global environment step")
        ax.set_ylabel("MSE TD loss")
        ax.set_title("Temporal-difference loss")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[2]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        for frame, label in ((dqn_episodes, "DQN"), (double_episodes, "Double DQN")):
            if frame is not None and "epsilon" in frame:
                axes[0].plot(frame["episode"], frame["epsilon"], linewidth=2, label=label)
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Epsilon")
        axes[0].set_title("Exploration schedule")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        for frame, label in ((dqn_steps, "DQN"), (double_steps, "Double DQN")):
            if frame is None:
                continue
            axes[1].plot(frame["global_step"], frame["avg_q"], linewidth=1, label=f"{label} mean chosen Q")
            axes[1].plot(frame["global_step"], frame["max_q"], linewidth=1, alpha=0.6, label=f"{label} max chosen Q")
        axes[1].set_xlabel("Global environment step")
        axes[1].set_ylabel("Q-value")
        axes[1].set_title("Q-value progression")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[3]:
        fig, ax = new_figure()
        for frame, label in ((dqn_steps, "DQN"), (double_steps, "Double DQN")):
            if frame is None:
                continue
            ax.plot(frame["global_step"], rolling(frame["grad_norm"]), linewidth=2, label=f"{label} grad norm (rolling 20)")
        ax.set_xlabel("Global environment step")
        ax.set_ylabel("L2 gradient norm")
        ax.set_title("Gradient norm")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tabs[4]:
        summary = read_csv(metrics_dir / "training_summary.csv")
        if summary is not None:
            columns = [
                column
                for column in (
                    "algorithm",
                    "seed",
                    "episodes",
                    "total_steps",
                    "total_optimization_steps",
                    "wall_time_s",
                    "final_avg_reward_50",
                    "mean_loss",
                    "mean_grad_norm",
                    "total_trainable_parameters",
                    "device",
                )
                if column in summary
            ]
            st.dataframe(summary[columns], width="stretch")

        if dqn_episodes is not None or double_episodes is not None:
            fig, ax = new_figure()
            for frame, label in ((dqn_episodes, "DQN"), (double_episodes, "Double DQN")):
                if frame is not None:
                    ax.plot(frame["episode"], frame["moving_avg_50"], linewidth=2, label=label)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Moving average reward (50)")
            ax.set_title("DQN vs Double DQN - learning curves")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    with tabs[5]:
        ablation = read_csv(PROJECT_ROOT / "results" / "evaluation" / "ablation_summary.csv")
        if ablation is None:
            ablation = read_csv(PROJECT_ROOT / "results" / "evaluation" / "ablation_results.csv")
        if ablation is None:
            st.info("No ablation results found. Run `python rl/experiments.py --mode ablation`.")
        else:
            st.dataframe(ablation, width="stretch")
            if "algorithm" in ablation and "eval_episode_reward_mean" in ablation:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                for ax, column, title in zip(
                    axes,
                    (
                        "eval_episode_reward_mean",
                        "eval_average_power_normalized_mean",
                        "eval_average_power_physical_mean",
                    ),
                    ("Evaluation reward", "Normalized power", "Physical power (W)"),
                ):
                    if column not in ablation:
                        continue
                    ax.bar(ablation["algorithm"], ablation[column], alpha=0.85)
                    ax.set_title(title)
                    ax.grid(True, axis="y", alpha=0.3)
                    ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    with tabs[6]:
        multi_seed = read_csv(PROJECT_ROOT / "results" / "evaluation" / "multi_seed_summary.csv")
        if multi_seed is None:
            multi_seed = read_csv(PROJECT_ROOT / "results" / "evaluation" / "multi_seed_results.csv")
        if multi_seed is None:
            st.info("No multi-seed results found. Run `python rl/experiments.py --mode multi_seed`.")
        else:
            st.dataframe(multi_seed, width="stretch")
            if "algorithm" in multi_seed and "eval_episode_reward_mean" in multi_seed:
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                specs = (
                    ("eval_episode_reward", "Evaluation reward"),
                    ("eval_average_power_normalized", "Normalized power"),
                    ("eval_average_power_physical", "Physical power (W)"),
                )
                for ax, (prefix, title) in zip(axes, specs):
                    mean_col = f"{prefix}_mean"
                    std_col = f"{prefix}_std"
                    if mean_col not in multi_seed:
                        continue
                    errors = multi_seed[std_col] if std_col in multi_seed else None
                    ax.bar(
                        multi_seed["algorithm"],
                        multi_seed[mean_col],
                        yerr=errors,
                        capsize=4,
                        alpha=0.85,
                    )
                    ax.set_title(title)
                    ax.grid(True, axis="y", alpha=0.3)
                    ax.tick_params(axis="x", rotation=20)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# ---------------------------------------------------------------------------
# Controller comparison
# ---------------------------------------------------------------------------

def render_comparison_section():
    st.header("Controller performance comparison")
    st.caption(
        "These are evaluation metrics on identical seeded wind traces, not training "
        "rewards. Both normalized and physical power are reported."
    )

    summary_path = Path(
        st.text_input(
            "Evaluation summary CSV",
            value=str(PROJECT_ROOT / "results" / "evaluation" / "controller_comparison_summary.csv"),
        )
    )
    frame = read_csv(summary_path)
    if frame is None:
        st.info(
            f"No evaluation summary found at {summary_path}. Run "
            "`python evaluation/agent_evaluation.py --q-episodes 500` first."
        )
    else:
        summary_metrics = st.columns(3)
        summary_metrics[0].metric("Controllers evaluated", str(len(frame)))
        if "average_power_physical_mean" in frame:
            peak = frame.loc[frame["average_power_physical_mean"].idxmax()]
            summary_metrics[1].metric("Highest mean physical power", f"{peak['average_power_physical_mean'] / 1000.0:.1f} kW")
            summary_metrics[2].metric("Controller at that value", str(peak.get("controller", "—")))
        display_columns = [
            column
            for column in (
                "controller",
                "episodes",
                "episode_reward_mean",
                "average_power_normalized_mean",
                "average_power_physical_mean",
                "mean_abs_misalignment_mean",
                "yaw_movement_frequency_mean",
                "control_stability_mean",
                "time_within_10_deg_mean",
                "inference_time_ms_mean",
            )
            if column in frame
        ]
        st.dataframe(frame[display_columns], width="stretch")

        if "controller" in frame:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            if "average_power_normalized_mean" in frame:
                axes[0].bar(frame["controller"], frame["average_power_normalized_mean"], alpha=0.85)
                axes[0].set_title("Average normalized power")
                axes[0].set_ylabel("Normalized power")
            if "average_power_physical_mean" in frame:
                axes[1].bar(frame["controller"], frame["average_power_physical_mean"] / 1000.0, alpha=0.85)
                axes[1].set_title("Average physical power")
                axes[1].set_ylabel("Power (kW)")
            for ax in axes:
                ax.grid(True, axis="y", alpha=0.3)
                ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            if "mean_abs_misalignment_mean" in frame:
                axes[0].bar(frame["controller"], frame["mean_abs_misalignment_mean"], alpha=0.85)
                axes[0].set_title("Mean absolute misalignment")
                axes[0].set_ylabel("Degrees")
            if "yaw_movement_frequency_mean" in frame:
                axes[1].bar(frame["controller"], frame["yaw_movement_frequency_mean"], alpha=0.85)
                axes[1].set_title("Yaw movement frequency")
                axes[1].set_ylabel("Fraction of steps")
            for ax in axes:
                ax.grid(True, axis="y", alpha=0.3)
                ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Legacy continuous Rule/PID comparison (preserved results)"):
        legacy_path = PROJECT_ROOT / "results" / "controller_comparison_summary.csv"
        legacy = read_csv(legacy_path)
        if legacy is None:
            st.info(f"Legacy summary not found: {legacy_path}")
        else:
            st.dataframe(legacy, width="stretch")
            st.caption(
                "These are the project's original results where Rule and PID apply "
                "continuous yaw commands. The new comparison projects their commands "
                "onto the discrete action space so that all controllers share the same "
                "environment interaction."
            )


def render_advanced_neural_section():
    """Render optional temporal/controlled-comparison artifacts without replacing the dashboard."""
    st.header("Advanced neural network experiments")
    st.caption("Sections appear only when their independently generated CSV artifacts are available.")
    base = PROJECT_ROOT / "results"
    imported_root = base / "colab_final_advanced_experiment"
    advanced_root = imported_root if imported_root.exists() else base
    st.caption(f"Reading advanced artifacts from: `{advanced_root}`")
    tabs = st.tabs(["LSTM-DQN", "Optimizers", "Activations", "Temporal attention", "Unified comparison"])
    with tabs[0]:
        st.markdown("**Architecture:** 20 causal observations → LSTM → linear three-Q-value head. "
                    "Episode resets discard history; initial observations are left-padded.")
        frame = read_csv(advanced_root / "temporal_experiments" / "evaluation" / "temporal_summary.csv")
        if frame is None: st.info("Run `python rl/temporal_experiments.py` to produce LSTM-DQN results.")
        else: st.dataframe(frame, width="stretch")
    with tabs[1]:
        frame = read_csv(advanced_root / "neural_comparisons" / "optimizer_summary.csv")
        if frame is None: st.info("Run `python rl/nn_comparisons.py --kind optimizer`.")
        else: st.dataframe(frame, width="stretch")
    with tabs[2]:
        frame = read_csv(advanced_root / "neural_comparisons" / "activation_summary.csv")
        if frame is None: st.info("Run `python rl/nn_comparisons.py --kind activation`.")
        else: st.dataframe(frame, width="stretch")
    with tabs[3]:
        attention = advanced_root / "temporal_experiments" / "attention"
        paths = sorted(attention.glob("*.csv")) if attention.exists() else []
        if not paths: st.info("Attention heatmaps appear after Attention-LSTM-DQN evaluation.")
        else:
            path = st.selectbox("Attention run", paths, format_func=lambda p: p.name)
            frame = read_csv(path)
            lag_cols = [c for c in frame.columns if c.startswith("lag_")] if frame is not None else []
            if frame is not None and lag_cols:
                fig, ax = plt.subplots(figsize=(10, 4)); image = ax.imshow(frame[lag_cols].head(100).T, aspect="auto", cmap="viridis")
                ax.set_xlabel("Decision step"); ax.set_ylabel("History position"); ax.set_title("Temporal attention weights")
                fig.colorbar(image, ax=ax, label="Softmax weight"); st.pyplot(fig); plt.close(fig)
    with tabs[4]:
        frame = read_csv(advanced_root / "unified_evaluation" / "unified_comparison.csv")
        if frame is None: st.info("Run `python evaluation/unified_comparison.py` after experiments complete.")
        else: st.dataframe(frame, width="stretch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Yaw Control Dashboard", layout="wide")
    header_left, header_right = st.columns([7, 1])
    with header_left:
        st.markdown(
            """
            <div class="dashboard-hero">
              <h1>Wind Turbine Yaw Control Dashboard</h1>
              <p>Neural-network training analysis and multi-controller evaluation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        dark_mode = st.toggle("Dark mode", key="dark_mode", help="Switch between light and dark analysis views.")
    apply_dashboard_theme(dark_mode)

    data_path_default = str(PROJECT_ROOT / "data" / "era5_chennai.csv")
    dqn_path_default = str(PROJECT_ROOT / "results" / "models" / "dqn.pth")
    double_path_default = str(PROJECT_ROOT / "results" / "models" / "double_dqn.pth")
    if not os.path.exists(dqn_path_default):
        dqn_path_default = str(PROJECT_ROOT / "rl" / "model.pth")

    models = {}
    temporal_models = {}
    device = torch.device("cpu")
    with st.expander("Run configuration", expanded=True):
        st.caption("Choose the wind dataset and optional model checkpoints used by the interactive trace.")
        data_path = st.text_input("Wind dataset path", value=data_path_default)
        col1, col2 = st.columns(2)
        with col1:
            dqn_path = st.text_input("DQN model path", value=dqn_path_default)
        with col2:
            double_path = st.text_input("Double DQN model path", value=double_path_default)

        if not os.path.exists(data_path):
            st.error(f"Dataset not found: {data_path}")
            st.info("Set a valid CSV containing wind_speed and wind_direction columns.")
            return

        for label, path in (("DQN", dqn_path), ("Double DQN", double_path)):
            if os.path.exists(path):
                try:
                    model, device, _ = load_model(path)
                    models[label] = model
                except Exception as exc:
                    st.warning(f"Could not load {label} model: {exc}")
            else:
                st.warning(f"{label} checkpoint not found: {path}")

        advanced_root = PROJECT_ROOT / "results" / "colab_final_advanced_experiment"
        if not advanced_root.exists():
            advanced_root = PROJECT_ROOT / "results"
        temporal_paths = {
            "LSTM-DQN": advanced_root / "temporal_experiments" / "models" / "lstm_dqn_seed0.pth",
            "Attention-LSTM-DQN": advanced_root / "temporal_experiments" / "models" / "attention_lstm_dqn_seed0.pth",
        }
        for label, path in temporal_paths.items():
            if not path.exists():
                continue
            try:
                temporal_models[label] = LSTMDQNAgent.load(path, device="cpu")
            except Exception as exc:
                st.warning(f"Could not load {label} model: {exc}")

        status_a, status_b, status_c = st.columns(3)
        status_a.metric("Wind data", "Ready")
        status_b.metric("Feedforward models", str(len(models)))
        status_c.metric("Temporal models", str(len(temporal_models)))

    with st.container(border=True):
        render_trace_section(models, temporal_models, device, data_path)
    st.divider()
    with st.container(border=True):
        render_training_section()
    st.divider()
    with st.container(border=True):
        render_advanced_neural_section()
    st.divider()
    with st.container(border=True):
        render_comparison_section()


if __name__ == "__main__":
    main()
