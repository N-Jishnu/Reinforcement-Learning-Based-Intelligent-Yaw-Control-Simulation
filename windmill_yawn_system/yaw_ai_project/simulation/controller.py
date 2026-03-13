import numpy as np
from config import YAW_ERROR_THRESHOLD

# ----------------------------------
# Utility Functions (CRITICAL)
# ----------------------------------

def angle_diff(a, b):
    """Shortest signed difference between two angles"""
    return (a - b + 180) % 360 - 180


def normalize_angle(angle):
    """Ensure angle stays within 0–360°"""
    return angle % 360


# ----------------------------------
# Simple Moving Average Controller
# ----------------------------------

class SimpleController:
    def __init__(self, averaging_window=30, error_threshold=YAW_ERROR_THRESHOLD):
        self.averaging_window = averaging_window
        self.error_threshold = error_threshold
        self.wind_direction_buffer = []

    def get_target_yaw(self, current_wind_direction, current_turbine_yaw):

        current_wind_direction = normalize_angle(current_wind_direction)

        # Add reading
        self.wind_direction_buffer.append(current_wind_direction)

        # Maintain buffer size
        if len(self.wind_direction_buffer) > self.averaging_window:
            self.wind_direction_buffer.pop(0)

        # Vector mean (correct for circular data)
        avg_wind_direction = self._calculate_vector_mean(self.wind_direction_buffer)

        error = angle_diff(avg_wind_direction, current_turbine_yaw)

        if abs(error) > self.error_threshold:
            return avg_wind_direction
        else:
            return current_turbine_yaw

    def _calculate_vector_mean(self, angles):
        angles_rad = np.radians(angles)
        x = np.mean(np.cos(angles_rad))
        y = np.mean(np.sin(angles_rad))
        mean_rad = np.arctan2(y, x)
        return normalize_angle(np.degrees(mean_rad))


# ----------------------------------
# PID Controller
# ----------------------------------

class PIDController:
    def __init__(
        self,
        Kp=0.5,
        Ki=0.01,
        Kd=0.1,
        dt=1.0,
        error_threshold=YAW_ERROR_THRESHOLD,
        max_correction=10.0  # ⭐ prevents crazy jumps
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.error_threshold = error_threshold
        self.max_correction = max_correction

        self.integral = 0.0
        self.prev_error = 0.0

    def get_target_yaw(self, current_wind_direction, current_turbine_yaw):

        current_wind_direction = normalize_angle(current_wind_direction)

        error = angle_diff(current_wind_direction, current_turbine_yaw)

        # Deadband
        if abs(error) < self.error_threshold:
            return current_turbine_yaw

        # PID
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        correction = (
            self.Kp * error
            + self.Ki * self.integral
            + self.Kd * derivative
        )

        # ⭐ Clamp correction (VERY IMPORTANT)
        correction = np.clip(correction, -self.max_correction, self.max_correction)

        self.prev_error = error

        return normalize_angle(current_turbine_yaw + correction)


# ----------------------------------
# Predictive Controller
# ----------------------------------

class PredictiveController:
    def __init__(self, lookahead_steps=5, history_len=10):
        self.lookahead = lookahead_steps
        self.history_len = history_len
        self.history = []

    def get_target_yaw(self, current_wind_direction, current_turbine_yaw):

        current_wind_direction = normalize_angle(current_wind_direction)

        self.history.append(current_wind_direction)

        if len(self.history) > self.history_len:
            self.history.pop(0)

        if len(self.history) < 3:
            return current_wind_direction

        # Unwrap angles
        angles_rad = np.radians(self.history)
        unwrapped = np.unwrap(angles_rad)

        # Linear regression
        x = np.arange(len(unwrapped))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, unwrapped, rcond=None)[0]

        future_x = (len(unwrapped) - 1) + self.lookahead
        predicted_rad = m * future_x + c

        predicted_deg = np.degrees(predicted_rad)

        return normalize_angle(predicted_deg)
