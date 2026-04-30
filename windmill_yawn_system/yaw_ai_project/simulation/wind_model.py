import numpy as np

from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import compute_power


def generate_wind_series(duration, dt, data_path):
    if dt != 1.0:
        raise ValueError("Time step must be fixed at 1 second (dt=1.0)")

    wind = WindField(duration=duration, dt=dt, data_path=data_path)
    steps = int(duration / dt)

    wind_speed = np.zeros(steps, dtype=float)
    wind_direction = np.zeros(steps, dtype=float)

    for t in range(steps):
        speed_t, direction_t = wind.get_wind_data(t)
        wind_speed[t] = speed_t
        wind_direction[t] = direction_t

    return wind_speed, wind_direction


def simulation_step(turbine, wind_speed, wind_direction, action_delta_deg, dt=1.0, mode="normalized"):
    if dt != 1.0:
        raise ValueError("Time step must be fixed at 1 second (dt=1.0)")

    target_yaw = (turbine.get_yaw_angle() + float(action_delta_deg)) % 360
    turbine.step(target_yaw, dt)

    yaw_angle = turbine.get_yaw_angle()
    misalignment = (wind_direction - yaw_angle + 180) % 360 - 180
    power = compute_power(wind_speed, misalignment, mode=mode)

    return yaw_angle, misalignment, power


def run_simulation_loop(actions, duration, dt, data_path, initial_yaw=180.0, mode="normalized"):
    wind_speed, wind_direction = generate_wind_series(duration, dt, data_path)
    turbine = TurbineModel(initial_yaw=initial_yaw)

    steps = len(wind_speed)
    yaw_angle = np.zeros(steps, dtype=float)
    misalignment = np.zeros(steps, dtype=float)
    power_output = np.zeros(steps, dtype=float)

    for t in range(steps):
        action = float(actions[t]) if t < len(actions) else 0.0
        yaw_t, mis_t, power_t = simulation_step(
            turbine=turbine,
            wind_speed=wind_speed[t],
            wind_direction=wind_direction[t],
            action_delta_deg=action,
            dt=dt,
            mode=mode,
        )
        yaw_angle[t] = yaw_t
        misalignment[t] = mis_t
        power_output[t] = power_t

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "yaw_angle": yaw_angle,
        "misalignment": misalignment,
        "power_output": power_output,
    }
