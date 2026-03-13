import numpy as np
from config import (
    AIR_DENSITY,
    ROTOR_AREA,
    POWER_COEFFICIENT_MAX,
    CUT_IN_SPEED,
    CUT_OUT_SPEED,
    RATED_SPEED,
    RATED_POWER
)


def calculate_power(wind_speed, wind_direction, turbine_yaw):
    """
    Calculate turbine electrical power output.

    Args:
        wind_speed (float): Wind speed (m/s)
        wind_direction (float): Wind direction (deg)
        turbine_yaw (float): Turbine yaw angle (deg)

    Returns:
        float: Power output (Watts)
    """

    # ----------------------------------
    # Cut-in / Cut-out Check
    # ----------------------------------
    if wind_speed < CUT_IN_SPEED or wind_speed > CUT_OUT_SPEED:
        return 0.0

    # ----------------------------------
    # Yaw Misalignment
    # ----------------------------------
    yaw_error = (wind_direction - turbine_yaw + 180) % 360 - 180
    yaw_error_rad = np.radians(yaw_error)

    # Power loss due to misalignment
    alignment_factor = np.cos(yaw_error_rad) ** 3

    # Prevent negative power (backwind)
    alignment_factor = max(alignment_factor, 0.0)

    # ----------------------------------
    # Aerodynamic Power
    # ----------------------------------
    if wind_speed < RATED_SPEED:
        # Region II → Cp max tracking
        power = (
            0.5
            * AIR_DENSITY
            * ROTOR_AREA
            * (wind_speed ** 3)
            * POWER_COEFFICIENT_MAX
        )
    else:
        # Region III → Rated power cap
        power = RATED_POWER

    # ----------------------------------
    # Apply Yaw Loss
    # ----------------------------------
    actual_power = power * alignment_factor

    return max(actual_power, 0.0)