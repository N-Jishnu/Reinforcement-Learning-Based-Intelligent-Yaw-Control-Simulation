import numpy as np

# ----------------------------------
# Wake Model Parameters
# ----------------------------------

WAKE_DECAY = 0.25       # Wake strength coefficient
WAKE_EXPANSION = 0.04   # Wake spreading factor
MIN_WIND_SPEED = 0.1


# ----------------------------------
# Jensen Wake Model
# ----------------------------------

def jensen_wake(upstream_speed, distance, rotor_diameter=120):
    """
    Compute wind speed reduction due to wake using
    a simplified Jensen wake model.

    Args:
        upstream_speed (float): incoming wind speed
        distance (float): distance between turbines (m)
        rotor_diameter (float): turbine rotor diameter

    Returns:
        float: reduced wind speed
    """

    wake_radius = rotor_diameter / 2 + WAKE_EXPANSION * distance

    deficit = (1 - np.sqrt(1 - WAKE_DECAY)) * (rotor_diameter / (2 * wake_radius))**2

    downstream_speed = upstream_speed * (1 - deficit)

    return max(downstream_speed, MIN_WIND_SPEED)


# ----------------------------------
# Wake Deflection From Yaw
# ----------------------------------

def yaw_deflection_factor(yaw_angle):
    """
    When turbine is yawed, wake is partially deflected.

    Args:
        yaw_angle (float): turbine yaw misalignment (degrees)

    Returns:
        float: wake reduction factor
    """

    yaw_rad = np.radians(yaw_angle)

    # cosine loss factor
    return np.cos(yaw_rad)**2


# ----------------------------------
# Apply Wake Between Turbines
# ----------------------------------

def apply_wake(upstream_speed, upstream_yaw, distance):
    """
    Combine Jensen wake + yaw deflection.

    Args:
        upstream_speed (float)
        upstream_yaw (float)
        distance (float)

    Returns:
        float: wind speed reaching next turbine
    """

    # basic wake
    speed_after_wake = jensen_wake(upstream_speed, distance)

    # yaw reduces wake strength
    deflection = yaw_deflection_factor(upstream_yaw)

    adjusted_speed = speed_after_wake * deflection

    return max(adjusted_speed, MIN_WIND_SPEED)