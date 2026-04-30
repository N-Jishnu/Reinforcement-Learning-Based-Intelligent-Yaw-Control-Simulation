import numpy as np


K_NORMALIZATION = 0.5
AIR_DENSITY = 1.225
POWER_COEFFICIENT = 0.4
ROTOR_RADIUS = 40.0
ROTOR_AREA = np.pi * (ROTOR_RADIUS ** 2)


def compute_power(wind_speed, misalignment, mode="normalized"):
    yaw_error_rad = np.radians(misalignment)
    alignment_factor = np.cos(yaw_error_rad) ** 3
    alignment_factor = max(alignment_factor, 0.0)

    if mode == "normalized":
        power = K_NORMALIZATION * (wind_speed ** 3) * alignment_factor
        return max(power, 0.0)

    if mode == "physical":
        power = (
            0.5
            * AIR_DENSITY
            * ROTOR_AREA
            * POWER_COEFFICIENT
            * (wind_speed ** 3)
            * alignment_factor
        )
        return max(power, 0.0)

    raise ValueError("mode must be 'normalized' or 'physical'")


def calculate_power(wind_speed, wind_direction, turbine_yaw, mode="normalized"):
    misalignment = (wind_direction - turbine_yaw + 180) % 360 - 180
    return compute_power(wind_speed, misalignment, mode=mode)
