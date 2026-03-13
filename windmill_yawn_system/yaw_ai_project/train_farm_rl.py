from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from simulation.farm_env import WindFarmEnv

env = DummyVecEnv([lambda: WindFarmEnv()])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
)

model.learn(total_timesteps=800000)

model.save("ppo_wind_farm")