import numpy as np
from config import (
    MAX_YAW_RATE,
    YAW_INERTIA,
    ACTUATION_DELAY
)


# ----------------------------------
# Utility Functions
# ----------------------------------

def angle_diff(a, b):
    """Shortest signed angular difference"""
    return (a - b + 180) % 360 - 180


def normalize_angle(angle):
    """Wrap angle to [0, 360)"""
    return angle % 360


# ----------------------------------
# Turbine Model
# ----------------------------------

class TurbineModel:

    def __init__(self, initial_yaw=0.0):

        self.yaw_angle = normalize_angle(initial_yaw)

        # ⭐ Dynamic states
        self.yaw_rate = 0.0

        # ⭐ Delay buffer
        self.command_buffer = [self.yaw_angle] * ACTUATION_DELAY

    # ----------------------------------
    # Main Dynamics Step
    # ----------------------------------

    def step(self, target_yaw, dt):

        target_yaw = normalize_angle(target_yaw)

        # ⭐ Apply actuation delay
        self.command_buffer.append(target_yaw)
        delayed_target = self.command_buffer.pop(0)

        # Compute shortest angular difference
        delta = angle_diff(delayed_target, self.yaw_angle)

        # ⭐ Desired yaw rate (limited)
        desired_rate = np.clip(delta / dt, -MAX_YAW_RATE, MAX_YAW_RATE)

        # ⭐ Apply inertia (low-pass filtering of yaw rate)
        self.yaw_rate = (
            (1 - YAW_INERTIA) * self.yaw_rate
            + YAW_INERTIA * desired_rate
        )

        # ⭐ Update yaw angle
        self.yaw_angle = normalize_angle(self.yaw_angle + self.yaw_rate * dt)

    # ----------------------------------
    # Accessors
    # ----------------------------------

    def get_yaw_angle(self):
        return self.yaw_angle

    def get_yaw_rate(self):
        return self.yaw_rate

    def set_yaw_angle(self, yaw):
        """Direct yaw setter (used by constrained RL env)."""
        self.yaw_angle = normalize_angle(yaw)
        self.yaw_rate = 0.0  # Reset rate for stability