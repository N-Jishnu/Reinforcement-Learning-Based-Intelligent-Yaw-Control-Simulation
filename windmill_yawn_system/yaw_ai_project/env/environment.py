import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.power_model import compute_power
from simulation.turbine_model import TurbineModel
from simulation.wind_model import generate_wind_series


MAX_STEPS = 1000
TIME_STEP = 1.0
WIND_SPEED_MIN = 3.0
WIND_SPEED_MAX = 15.0
YAW_ACTION_STEP_DEG = 2.0
POWER_MAX_NORMALIZED_MODE = 0.5 * (WIND_SPEED_MAX ** 3)


def _wrap_360(angle_deg):
    return angle_deg % 360.0


def _angle_diff(a_deg, b_deg):
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


class YawRLEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data_path="data/era5_chennai.csv"):
        super().__init__()
        self.data_path = data_path

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        self.wind_speed_series = None
        self.wind_direction_series = None
        self.turbine = None
        self.current_step = 0
        self.prev_wind_direction = 0.0
        self.prev_action_dir = 0
        self.prev_yaw_change = 0.0

    def _state(self):
        idx = self.current_step % len(self.wind_speed_series)
        wind_speed = float(self.wind_speed_series[idx])
        wind_direction = float(self.wind_direction_series[idx])
        yaw_angle = float(self.turbine.get_yaw_angle())
        misalignment = _angle_diff(wind_direction, yaw_angle)
        wind_direction_change = _angle_diff(wind_direction, self.prev_wind_direction)

        state = np.array(
            [
                wind_direction / 360.0,
                (wind_speed - WIND_SPEED_MIN) / (WIND_SPEED_MAX - WIND_SPEED_MIN),
                yaw_angle / 360.0,
                (misalignment + 180.0) / 360.0,
                (wind_direction_change + 180.0) / 360.0,
            ],
            dtype=np.float32,
        )
        return np.clip(state, 0.0, 1.0), wind_speed, wind_direction, misalignment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.wind_speed_series, self.wind_direction_series = generate_wind_series(
            duration=MAX_STEPS,
            dt=TIME_STEP,
            data_path=self.data_path,
        )

        self.turbine = TurbineModel(initial_yaw=0.0)
        self.current_step = 0
        self.prev_action_dir = 0
        self.prev_wind_direction = float(self.wind_direction_series[0])
        self.prev_yaw_change = 0.0

        obs, _, _, _ = self._state()
        return obs, {}

    def step(self, action):
        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError("Action must be 0 (left), 1 (right), or 2 (hold)")

        action_dir = -1 if action == 0 else (1 if action == 1 else 0)
        yaw_change_command = action_dir * YAW_ACTION_STEP_DEG

        current_yaw = self.turbine.get_yaw_angle()
        target_yaw = _wrap_360(current_yaw + yaw_change_command)
        self.turbine.step(target_yaw, TIME_STEP)
        yaw_after = self.turbine.get_yaw_angle()
        yaw_change_applied = _angle_diff(yaw_after, current_yaw)

        obs, wind_speed, wind_direction, misalignment = self._state()

        power = compute_power(wind_speed, misalignment, mode="normalized")
        power_normalized = power / POWER_MAX_NORMALIZED_MODE

        alignment_term = max(np.cos(np.radians(misalignment)) ** 3, 0.0)

        misalignment_norm = abs(misalignment) / 180.0
        misalignment_penalty = 0.60 * (misalignment_norm ** 2)

        switched_direction = 0
        if action_dir != 0 and self.prev_action_dir != 0 and action_dir != self.prev_action_dir:
            switched_direction = 1

        switch_penalty = 0.15 * switched_direction

        rapid_flip_penalty = 0.0
        if switched_direction and abs(getattr(self, "prev_yaw_change", 0.0)) > 0.0:
            rapid_flip_penalty = 0.05

        movement_penalty = 0.06 * abs(yaw_change_applied)

        alignment_bonus = 0.15 if abs(misalignment) <= 5.0 else 0.0

        reward = (
            1.60 * power_normalized
            + 0.55 * alignment_term
            - 0.50 * (misalignment_norm ** 2)
            - 0.05 * abs(yaw_change_applied)
            - 0.18 * switched_direction
            - rapid_flip_penalty
            + alignment_bonus
        )

        self.prev_action_dir = action_dir
        self.prev_wind_direction = wind_direction
        self.prev_yaw_change = yaw_change_applied
        self.current_step += 1

        terminated = self.current_step >= MAX_STEPS
        truncated = False
        info = {
            "power_normalized": float(power_normalized),
            "power": float(power),
            "misalignment": float(misalignment),
            "yaw_change": float(yaw_change_applied),
            "switched_direction": int(switched_direction),
            "alignment_term": float(alignment_term),
            "alignment_bonus": float(alignment_bonus),
            "rapid_flip_penalty": float(rapid_flip_penalty),
        }
        return obs, float(reward), terminated, truncated, info
