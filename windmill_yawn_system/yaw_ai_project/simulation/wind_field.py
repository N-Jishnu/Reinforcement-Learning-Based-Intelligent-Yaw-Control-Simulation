import numpy as np
import pandas as pd


class WindField:

    def __init__(
        self,
        duration,
        dt,
        data_path,
        turbulence_intensity=0.02,
        gust_probability=0.008,
        gust_strength=(4.0, 10.0),
        shift_probability=0.003
    ):
        """
        High-fidelity wind model using:
        • Real dataset
        • Turbulence
        • Gust events
        • Directional shifts
        """

        self.duration = duration
        self.dt = dt
        self.requested_steps = int(duration / dt)

        self._load_real_data(data_path)

        self.wind_speeds = np.array(self.wind_speeds, dtype=float)
        self.wind_directions = np.array(self.wind_directions, dtype=float) % 360

        self.time_steps = len(self.wind_speeds)
        self.time = np.arange(0, self.requested_steps * dt, dt)

        # Disturbance parameters
        self.turbulence_intensity = turbulence_intensity
        self.gust_probability = gust_probability
        self.gust_strength = gust_strength
        self.shift_probability = shift_probability

        # Internal disturbance states
        self.gust_remaining = 0
        self.gust_strength_active = 0
        self.gust_duration = 0

        self.direction_shift = 0.0

    # ----------------------------------
    # Load Dataset
    # ----------------------------------

    def _load_real_data(self, path):

        df = pd.read_csv(path)

        required_cols = {"wind_speed", "wind_direction"}

        if not required_cols.issubset(df.columns):
            raise ValueError(
                "CSV must contain columns: wind_speed, wind_direction"
            )

        speeds = df["wind_speed"].values.astype(float)
        directions = df["wind_direction"].values.astype(float) % 360

        if len(speeds) < 50:
            raise ValueError("Dataset too short.")

        self.wind_speeds = speeds
        self.wind_directions = directions

    # ----------------------------------
    # Turbulence Model
    # ----------------------------------

    def _apply_turbulence(self, speed, direction):

        speed_noise = np.random.normal(
            0,
            self.turbulence_intensity * speed
        )

        dir_noise = np.random.normal(
            0,
            self.turbulence_intensity * 20
        )

        speed = max(speed + speed_noise, 0.1)
        direction = (direction + dir_noise) % 360

        return speed, direction

    # ----------------------------------
    # Gust Model
    # ----------------------------------

    def _apply_gusts(self, speed):

        if self.gust_remaining > 0:

            progress = 1 - (self.gust_remaining / self.gust_duration)

            gust_profile = np.sin(progress * np.pi)

            speed += self.gust_strength_active * gust_profile

            self.gust_remaining -= 1

        else:

            if np.random.rand() < self.gust_probability:

                self.gust_strength_active = np.random.uniform(
                    *self.gust_strength
                )

                self.gust_duration = np.random.randint(10, 40)

                self.gust_remaining = self.gust_duration

        return speed

    # ----------------------------------
    # Directional Shift Model
    # ----------------------------------

    def _apply_direction_shifts(self, direction):

        if np.random.rand() < self.shift_probability:

            shift = np.random.choice([-1, 1]) * np.random.uniform(15, 35)

            self.direction_shift += shift

        # decay shift slowly (prevents drift explosion)
        self.direction_shift *= 0.999

        direction = (direction + self.direction_shift) % 360

        return direction

    # ----------------------------------
    # Data Access
    # ----------------------------------

    def get_wind_data(self, t_idx):

        base_idx = t_idx % self.time_steps

        speed = self.wind_speeds[base_idx]
        direction = self.wind_directions[base_idx]

        # Apply disturbances
        speed, direction = self._apply_turbulence(speed, direction)

        speed = self._apply_gusts(speed)

        direction = self._apply_direction_shifts(direction)

        # Clip unrealistic values
        speed = np.clip(speed, 0.1, 30)

        return float(speed), float(direction)