import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulation.wind_field import WindField
from simulation.turbine_model import TurbineModel
from simulation.power_model import calculate_power
from config import SIMULATION_DURATION, TIME_STEP


# ----------------------------------
# Wake Model
# ----------------------------------

WAKE_DECAY = 0.25
TURBINE_DISTANCE = 400


def apply_wake(speed, upstream_yaw, distance):

    yaw_rad = np.radians(upstream_yaw)

    wake_loss = WAKE_DECAY * np.cos(yaw_rad)

    wake_loss *= np.exp(-distance / 500)

    return max(speed * (1 - wake_loss), 0.1)


# ----------------------------------
# Utility
# ----------------------------------

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180


def normalize_angle(a):
    return a % 360


# ----------------------------------
# RL Environment
# ----------------------------------

class WindFarmEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # 3 turbines → 3 actions
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # observations
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP,
            data_path="data/era5_chennai.csv"
        )

        self.turbines = [
            TurbineModel(initial_yaw=180),
            TurbineModel(initial_yaw=180),
            TurbineModel(initial_yaw=180)
        ]

        self.current_step = 0

    # ----------------------------------

    def _build_obs(self, speed, direction):

        yaws = [t.get_yaw_angle() for t in self.turbines]

        return np.array([
            speed / 25,
            direction / 360,
            yaws[0] / 360,
            yaws[1] / 360,
            yaws[2] / 360
        ], dtype=np.float32)

    # ----------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.wind = WindField(
            SIMULATION_DURATION,
            TIME_STEP,
            data_path="data/era5_chennai.csv"
        )

        self.turbines = [
            TurbineModel(initial_yaw=180),
            TurbineModel(initial_yaw=180),
            TurbineModel(initial_yaw=180)
        ]

        self.current_step = 0

        speed, direction = self.wind.get_wind_data(0)

        obs = self._build_obs(speed, direction)

        return obs, {}

    # ----------------------------------

    def step(self, action):

        self.current_step += 1

        speed, direction = self.wind.get_wind_data(self.current_step)

        yaw_cmds = action - 1

        # --- Turbine 1 (no wake)

        yaw1 = self.turbines[0].get_yaw_angle()
        yaw1 = normalize_angle(yaw1 + yaw_cmds[0] * 2)
        self.turbines[0].step(yaw1, TIME_STEP)

        speed1 = speed

        # --- Turbine 2 (wake from T1)

        yaw2 = self.turbines[1].get_yaw_angle()
        yaw2 = normalize_angle(yaw2 + yaw_cmds[1] * 2)
        self.turbines[1].step(yaw2, TIME_STEP)

        speed2 = apply_wake(speed1, yaw1, TURBINE_DISTANCE)

        # --- Turbine 3 (wake from T2)

        yaw3 = self.turbines[2].get_yaw_angle()
        yaw3 = normalize_angle(yaw3 + yaw_cmds[2] * 2)
        self.turbines[2].step(yaw3, TIME_STEP)

        speed3 = apply_wake(speed2, yaw2, TURBINE_DISTANCE)

        # --- Power

        power1 = calculate_power(speed1, direction, yaw1)
        power2 = calculate_power(speed2, direction, yaw2)
        power3 = calculate_power(speed3, direction, yaw3)

        total_power = power1 + power2 + power3

        reward = total_power / 1e6

        terminated = self.current_step >= 1000
        truncated = False

        obs = self._build_obs(speed, direction)

        info = {
            "farm_power": total_power,
            "power_t1": power1,
            "power_t2": power2,
            "power_t3": power3
        }

        return obs, reward, terminated, truncated, info