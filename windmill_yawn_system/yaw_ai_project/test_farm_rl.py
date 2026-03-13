from stable_baselines3 import PPO
from simulation.farm_env import WindFarmEnv

model = PPO.load("ppo_wind_farm")

env = WindFarmEnv()

obs,_ = env.reset()

for step in range(50):

    action,_ = model.predict(obs)

    obs,reward,done,trunc,info = env.step(action)

    print("Farm Power:", info["farm_power"])