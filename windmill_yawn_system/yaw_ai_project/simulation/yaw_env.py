import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import calculate_power
from config import SIMULATION_DURATION, TIME_STEP

# ----------------------------------
# Constants
# ----------------------------------

MAX_STEPS = 1000
POWER_SCALE = 1e6

YAW_ERROR_WEIGHT = 0.0004
YAW_RATE_WEIGHT = 0.00005
CONTROL_EFFORT_WEIGHT = 0.005

MAX_YAW_RATE = 0.8
ACTUATION_DELAY = 3
YAW_TIME_CONSTANT = 5.0


# ----------------------------------
# Utility Functions
# ----------------------------------

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def normalize_angle(angle):
    return angle % 360


# ----------------------------------
# RL Environment
# ----------------------------------

class YawControlEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        # 8 observation features (all normalized)
        self.observation_space = spaces.Box(
            low=np.array([0]*8, dtype=np.float32),
            high=np.array([1]*8, dtype=np.float32),
            dtype=np.float32
        )

        self.wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP,
            data_path="data/era5_chennai.csv"
        )

        self.turbine = TurbineModel(initial_yaw=180)

        self.current_step = 0
        self.prev_yaw = None

        self.action_buffer = deque(maxlen=ACTUATION_DELAY)
        self.yaw_velocity = 0.0

        # ⭐ Wind history buffers
        self.wind_speed_hist = deque(maxlen=3)
        self.wind_dir_hist = deque(maxlen=3)

    # ----------------------------------
    # Build Observation
    # ----------------------------------

    def _build_obs(self, wind_speed, wind_dir, yaw_angle, yaw_error):

        speeds = list(self.wind_speed_hist)
        dirs = list(self.wind_dir_hist)

        obs = np.array([
            speeds[0] / 25.0,
            speeds[1] / 25.0,
            speeds[2] / 25.0,
            dirs[0] / 360.0,
            dirs[1] / 360.0,
            dirs[2] / 360.0,
            yaw_angle / 360.0,
            yaw_error / 180.0
        ], dtype=np.float32)

        return obs

    # ----------------------------------
    # Reset
    # ----------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP,
            data_path="data/era5_chennai.csv"
        )

        self.turbine = TurbineModel(initial_yaw=180)

        self.current_step = 0
        self.prev_yaw = self.turbine.get_yaw_angle()

        self.action_buffer.clear()
        for _ in range(ACTUATION_DELAY):
            self.action_buffer.append(1)

        self.yaw_velocity = 0.0

        wind_speed, wind_dir = self.wind.get_wind_data(0)

        # Initialize wind history
        self.wind_speed_hist.clear()
        self.wind_dir_hist.clear()

        for _ in range(3):
            self.wind_speed_hist.append(wind_speed)
            self.wind_dir_hist.append(wind_dir)

        yaw_angle = self.turbine.get_yaw_angle()
        yaw_error = angle_diff(wind_dir, yaw_angle)

        obs = self._build_obs(
            wind_speed,
            wind_dir,
            yaw_angle,
            yaw_error
        )

        return obs, {}

    # ----------------------------------
    # Step
    # ----------------------------------

    def step(self, action):

        self.current_step += 1

        self.action_buffer.append(action)
        delayed_action = self.action_buffer[0]

        yaw_cmd = delayed_action - 1

        idx = self.current_step % self.wind.time_steps
        wind_speed, wind_dir = self.wind.get_wind_data(idx)

        # Update wind history
        self.wind_speed_hist.append(wind_speed)
        self.wind_dir_hist.append(wind_dir)

        current_yaw = self.turbine.get_yaw_angle()

        desired_rate = yaw_cmd * MAX_YAW_RATE

        self.yaw_velocity += (
            (desired_rate - self.yaw_velocity)
            * TIME_STEP / YAW_TIME_CONSTANT
        )

        self.yaw_velocity = np.clip(
            self.yaw_velocity,
            -MAX_YAW_RATE,
            MAX_YAW_RATE
        )

        new_yaw = normalize_angle(
            current_yaw + self.yaw_velocity * TIME_STEP
        )

        self.turbine.step(new_yaw, TIME_STEP)

        yaw_angle = self.turbine.get_yaw_angle()
        yaw_error = angle_diff(wind_dir, yaw_angle)

        yaw_rate = angle_diff(yaw_angle, self.prev_yaw)
        self.prev_yaw = yaw_angle

        power = calculate_power(wind_speed, wind_dir, yaw_angle)

        # ----------------------------------
        # Reward Function
        # ----------------------------------

        alignment_term = np.cos(np.radians(yaw_error)) ** 3
        alignment_term = max(alignment_term, 0.0)

        scaled_power = power / POWER_SCALE

        yaw_error_penalty = YAW_ERROR_WEIGHT * (yaw_error ** 2)
        yaw_rate_penalty = YAW_RATE_WEIGHT * abs(yaw_rate)
        control_penalty = CONTROL_EFFORT_WEIGHT * abs(yaw_cmd)

        reward = (
            4.0 * alignment_term
            + scaled_power
            - yaw_error_penalty
            - yaw_rate_penalty
            - control_penalty
        )

        terminated = self.current_step >= MAX_STEPS
        truncated = False

        obs = self._build_obs(
            wind_speed,
            wind_dir,
            yaw_angle,
            yaw_error
        )

        info = {
            "power": power,
            "yaw_error": yaw_error,
            "yaw_rate": yaw_rate,
            "yaw_velocity": self.yaw_velocity
        }

        return obs, reward, terminated, truncated, info