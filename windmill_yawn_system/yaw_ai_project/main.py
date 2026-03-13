import sys
import os

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

from config import SIMULATION_DURATION, TIME_STEP
from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import calculate_power
from utils.metrics import (
    calculate_energy_production,
    calculate_yaw_error_stats
)
from simulation.controller import (
    SimpleController,
    PIDController,
    PredictiveController
)

# -----------------------------
# Configuration
# -----------------------------

USE_ERA5_DATA = True   # ⭐ Toggle real vs synthetic
ERA5_PATH = "data/era5_wind.csv"


# -----------------------------
# Utility Functions
# -----------------------------

def angle_diff(a, b):
    """Returns shortest signed angle difference"""
    return (a - b + 180) % 360 - 180


# -----------------------------
# Simulation Runner
# -----------------------------

def run_simulation():
    print("Initializing Simulation...")

    np.random.seed(42)

    # Ensure results folder exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1️⃣ Initialize Wind Field
    if USE_ERA5_DATA:
        print("Using ERA5 Real Wind Dataset ⭐")
        wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP,
            data_path=ERA5_PATH
        )
    else:
        print("Using Synthetic Wind Model")
        wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP
        )

    scenarios = []

    initial_yaw = wind.wind_directions[0]

    # 2️⃣ Simple Controller
    print("Running scenario: Simple Controller")
    scenarios.append({
        'name': 'Simple (MA)',
        'data': run_scenario(
            wind,
            TurbineModel(initial_yaw=initial_yaw),
            SimpleController(averaging_window=30),
            "Simple"
        )
    })

    # 3️⃣ PID Controller
    print("Running scenario: PID Controller")
    scenarios.append({
        'name': 'PID',
        'data': run_scenario(
            wind,
            TurbineModel(initial_yaw=initial_yaw),
            PIDController(Kp=0.8, Ki=0.001, Kd=0.2, dt=TIME_STEP),
            "PID"
        )
    })

    # 4️⃣ Predictive Controller
    print("Running scenario: Predictive Controller")
    scenarios.append({
        'name': 'Predictive (Reg)',
        'data': run_scenario(
            wind,
            TurbineModel(initial_yaw=initial_yaw),
            PredictiveController(lookahead_steps=10, history_len=15),
            "Predictive"
        )
    })

    # 5️⃣ Perfect Controller (Baseline)
    print("Running scenario: Perfect Controller")

    class PerfectController:
        def get_target_yaw(self, current_wind, current_yaw):
            return current_wind

    scenarios.append({
        'name': 'Perfect',
        'data': run_scenario(
            wind,
            TurbineModel(initial_yaw=initial_yaw),
            PerfectController(),
            "Perfect"
        )
    })

    # 6️⃣ Analysis
    print("\n--- Comparative Results ---")
    for s in scenarios:
        analyze_results(s['data'], s['name'])

    # 7️⃣ Visualization
    plot_all_scenarios(scenarios)


# -----------------------------
# Scenario Execution
# -----------------------------

def run_scenario(wind, turbine, controller, name):

    history = {
        'time': wind.time,
        'wind_speed': wind.wind_speeds,
        'wind_direction': wind.wind_directions,
        'turbine_yaw': [],
        'power_output': []
    }

    for t_idx in range(wind.time_steps):

        true_speed, true_direction = wind.get_wind_data(t_idx)

        # Sensor noise
        if name == "Perfect":
            measured_direction = true_direction
        else:
            measured_direction = true_direction + np.random.normal(0, 2.0)

        measured_direction %= 360

        target_yaw = controller.get_target_yaw(
            measured_direction,
            turbine.get_yaw_angle()
        )

        turbine.step(target_yaw, TIME_STEP)

        current_yaw = turbine.get_yaw_angle()

        power = calculate_power(true_speed, true_direction, current_yaw)

        history['turbine_yaw'].append(current_yaw)
        history['power_output'].append(power)

    return history


# -----------------------------
# Analysis
# -----------------------------

def analyze_results(history, label):

    total_energy = calculate_energy_production(
        history['power_output'],
        TIME_STEP
    )

    yaw_stats = calculate_yaw_error_stats(
        history['wind_direction'],
        history['turbine_yaw']
    )

    print(f"[{label}]")
    print(f"  Total Energy: {total_energy/1e6:.2f} MJ")
    print(f"  Mean Yaw Error: {yaw_stats['mean_absolute_error']:.2f}°")


# -----------------------------
# Plotting
# -----------------------------

def plot_all_scenarios(scenarios):

    time = scenarios[0]['data']['time']
    baseline_energy = np.cumsum(
        scenarios[-1]['data']['power_output']
    ) * TIME_STEP

    plt.figure(figsize=(14, 12))

    # 1️⃣ Yaw Tracking
    plt.subplot(3, 1, 1)
    plt.plot(time, scenarios[0]['data']['wind_direction'],
             label='Wind (Ref)', color='black', alpha=0.3)

    for s in scenarios:
        linestyle = '--' if s['name'] == 'Perfect' else '-'
        plt.plot(time, s['data']['turbine_yaw'],
                 label=s['name'], linestyle=linestyle)

    plt.ylabel("Angle (deg)")
    plt.title("Yaw Tracking")
    plt.legend()
    plt.grid(True)

    # 2️⃣ Energy
    plt.subplot(3, 1, 2)
    for s in scenarios:
        e = np.cumsum(s['data']['power_output']) * TIME_STEP / 1e6
        plt.plot(time, e, label=s['name'])

    plt.ylabel("Energy (MJ)")
    plt.title("Energy Production")
    plt.legend()
    plt.grid(True)

    # 3️⃣ Efficiency
    plt.subplot(3, 1, 3)
    baseline_safe = np.where(baseline_energy > 0, baseline_energy, 1.0)

    for s in scenarios[:-1]:
        e = np.cumsum(s['data']['power_output']) * TIME_STEP
        efficiency = e / baseline_safe * 100
        plt.plot(time, efficiency, label=s['name'])

    plt.ylabel("% Efficiency")
    plt.xlabel("Time (s)")
    plt.title("Efficiency vs Perfect")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__),
                               "results", "final_comparison.png")

    plt.savefig(output_path)
    print(f"\nPlot saved → {output_path}")
    plt.show()


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    run_simulation()