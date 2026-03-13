import numpy as np

def calculate_energy_production(power_series, dt):
    """
    Calculate total energy produced.
    
    Args:
        power_series (list or np.array): Time series of power output (Watts).
        dt (float): Time step (seconds).
        
    Returns:
        float: Total energy in Joules (or Watt-seconds).
    """
    return np.sum(power_series) * dt

def calculate_yaw_error_stats(wind_directions, turbine_yaws):
    """
    Calculate statistics about yaw misalignment.
    
    Returns:
        dict: Mean absolute error, max error, etc.
    """
    wd = np.array(wind_directions)
    ty = np.array(turbine_yaws)
    
    errors = (wd - ty + 180) % 360 - 180
    abs_errors = np.abs(errors)
    
    return {
        "mean_absolute_error": np.mean(abs_errors),
        "std_error": np.std(abs_errors),
        "max_error": np.max(abs_errors)
    }
