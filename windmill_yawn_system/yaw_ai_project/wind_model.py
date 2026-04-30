import numpy as np

from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import calculate_power


def generate_wind_series(duration, dt, data_path):
    wind = WindField(duration=duration, dt=dt, data_path=data_path)
    steps = int(duration / dt)

    wind_speed = np.zeros(steps, dtype=float)
    wind_direction = np.zeros(steps, dtype=float)

    for t in range(steps):
        speed_t, direction_t = wind.get_wind_data(t)
        wind_speed[t] = speed_t
        wind_direction[t] = direction_t

    return wind_speed, wind_direction


def run_simulation_loop(actions, duration, dt, data_path, initial_yaw=180.0):
    wind_speed, wind_direction = generate_wind_series(duration, dt, data_path)
    turbine = TurbineModel(initial_yaw=initial_yaw)

    steps = len(wind_speed)
    yaw_angle = np.zeros(steps, dtype=float)
    misalignment = np.zeros(steps, dtype=float)
    power_output = np.zeros(steps, dtype=float)

    max_actions = min(steps, len(actions))
    for t in range(max_actions):
        action = float(actions[t])
        target = (turbine.get_yaw_angle() + action) % 360
        turbine.step(target, dt)

        yaw_t = turbine.get_yaw_angle()
        yaw_angle[t] = yaw_t

        misalignment_t = (wind_direction[t] - yaw_t + 180) % 360 - 180
        misalignment[t] = misalignment_t

        power_output[t] = calculate_power(wind_speed[t], wind_direction[t], yaw_t)

    if max_actions < steps:
        yaw_angle[max_actions:] = yaw_angle[max_actions - 1] if max_actions > 0 else initial_yaw
        for t in range(max_actions, steps):
            misalignment_t = (wind_direction[t] - yaw_angle[t] + 180) % 360 - 180
            misalignment[t] = misalignment_t
            power_output[t] = calculate_power(wind_speed[t], wind_direction[t], yaw_angle[t])

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "yaw_angle": yaw_angle,
        "misalignment": misalignment,
        "power_output": power_output,
    }
