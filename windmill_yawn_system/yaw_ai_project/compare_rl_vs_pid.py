import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import os

from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import calculate_power
from simulation.controller import PIDController
from simulation.yaw_env import angle_diff
from utils.metrics import (
    calculate_energy_production,
    calculate_yaw_error_stats
)
from config import SIMULATION_DURATION, TIME_STEP

DATA_PATH = "data/era5_chennai.csv"


# ----------------------------------
# Helper Metrics
# ----------------------------------

def calculate_control_effort(yaw_history):
    movement = 0.0
    for i in range(1, len(yaw_history)):
        movement += abs(angle_diff(yaw_history[i], yaw_history[i - 1]))
    return movement


# ----------------------------------
# PID Simulation
# ----------------------------------

def run_pid_simulation(wind):

    turbine = TurbineModel(initial_yaw=wind.wind_directions[0])
    controller = PIDController(Kp=0.8, Ki=0.001, Kd=0.2, dt=TIME_STEP)

    yaw_history = []
    power_history = []

    for t in range(wind.requested_steps):

        speed, direction = wind.get_wind_data(t)

        target_yaw = controller.get_target_yaw(
            direction,
            turbine.get_yaw_angle()
        )

        turbine.step(target_yaw, TIME_STEP)

        yaw = turbine.get_yaw_angle()
        power = calculate_power(speed, direction, yaw)

        yaw_history.append(yaw)
        power_history.append(power)

    return yaw_history, power_history


# ----------------------------------
# RL Simulation (Wind history aware)
# ----------------------------------

def run_rl_simulation(wind):

    model = PPO.load("ppo_yaw_controller_era5", device="cpu")

    turbine = TurbineModel(initial_yaw=wind.wind_directions[0])

    yaw_history = []
    power_history = []

    yaw_velocity = 0.0
    MAX_YAW_RATE = 0.8
    YAW_TIME_CONSTANT = 5.0

    # Wind history buffers
    speed_hist = [wind.wind_speeds[0]] * 3
    dir_hist = [wind.wind_directions[0]] * 3

    for t in range(wind.requested_steps):

        speed, direction = wind.get_wind_data(t)

        # Update history
        speed_hist.pop(0)
        speed_hist.append(speed)

        dir_hist.pop(0)
        dir_hist.append(direction)

        yaw = turbine.get_yaw_angle()
        yaw_error = angle_diff(direction, yaw)

        # Build normalized observation (8 features)
        obs = np.array([
            speed_hist[0] / 25,
            speed_hist[1] / 25,
            speed_hist[2] / 25,
            dir_hist[0] / 360,
            dir_hist[1] / 360,
            dir_hist[2] / 360,
            yaw / 360,
            yaw_error / 180
        ], dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)

        yaw_cmd = action - 1
        desired_rate = yaw_cmd * MAX_YAW_RATE

        yaw_velocity += (
            (desired_rate - yaw_velocity)
            * TIME_STEP / YAW_TIME_CONSTANT
        )

        yaw_velocity = np.clip(
            yaw_velocity,
            -MAX_YAW_RATE,
            MAX_YAW_RATE
        )

        new_yaw = (yaw + yaw_velocity * TIME_STEP) % 360

        turbine.step(new_yaw, TIME_STEP)

        yaw = turbine.get_yaw_angle()
        power = calculate_power(speed, direction, yaw)

        yaw_history.append(yaw)
        power_history.append(power)

    return yaw_history, power_history


# ----------------------------------
# Evaluation
# ----------------------------------

def evaluate(name, wind, yaw_hist, power_hist):

    energy = calculate_energy_production(power_hist, TIME_STEP)

    wind_dirs = [wind.get_wind_data(t)[1] for t in range(len(yaw_hist))]

    yaw_stats = calculate_yaw_error_stats(
        wind_dirs,
        yaw_hist
    )

    effort = calculate_control_effort(yaw_hist)

    print(f"\n[{name}]")
    print(f"Total Energy: {energy/1e6:.2f} MJ")
    print(f"Mean Yaw Error: {yaw_stats['mean_absolute_error']:.2f}°")
    print(f"Control Effort: {effort:.2f}° total yaw movement")

    return energy, yaw_stats, effort


# ----------------------------------
# Plotting
# ----------------------------------

def plot_comparison(wind, pid_yaw, rl_yaw, pid_power, rl_power):

    time = wind.time[:len(pid_yaw)]
    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(14, 10))

    wind_dirs = [wind.get_wind_data(t)[1] for t in range(len(pid_yaw))]

    plt.subplot(2, 1, 1)

    plt.plot(time, wind_dirs, label="Wind Direction", alpha=0.3)
    plt.plot(time, pid_yaw, label="PID Yaw")
    plt.plot(time, rl_yaw, label="RL Yaw", linestyle="--")

    plt.ylabel("Angle (deg)")
    plt.title("Yaw Tracking Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(time, np.array(pid_power)/1e6, label="PID Power")
    plt.plot(time, np.array(rl_power)/1e6, label="RL Power", linestyle="--")

    plt.ylabel("Power (MW)")
    plt.xlabel("Time (s)")
    plt.title("Power Output Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/rl_vs_pid.png")

    print("\nPlot saved → results/rl_vs_pid.png")
    plt.show()


# ----------------------------------
# Main
# ----------------------------------

def main():

    print("Running RL vs PID Comparison (ERA5 Real Wind)...")

    np.random.seed(42)

    wind = WindField(
        SIMULATION_DURATION,
        TIME_STEP,
        data_path=DATA_PATH
    )

    pid_yaw, pid_power = run_pid_simulation(wind)
    rl_yaw, rl_power = run_rl_simulation(wind)

    pid_energy, pid_stats, pid_effort = evaluate(
        "PID Controller", wind, pid_yaw, pid_power
    )

    rl_energy, rl_stats, rl_effort = evaluate(
        "RL PPO Controller", wind, rl_yaw, rl_power
    )

    efficiency = (rl_energy / pid_energy) * 100

    print("\n--- Comparative Summary ---")
    print(f"RL Energy Efficiency vs PID: {efficiency:.2f}%")

    if rl_effort < pid_effort:
        print("RL uses smoother control (less mechanical wear) ✅")
    else:
        print("PID uses less control effort")

    plot_comparison(wind, pid_yaw, rl_yaw, pid_power, rl_power)


if __name__ == "__main__":
    main()