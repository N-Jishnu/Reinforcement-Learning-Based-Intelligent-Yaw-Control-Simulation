from stable_baselines3 import PPO
from simulation.yaw_env import YawControlEnv

print("Loading trained PPO model...")

model = PPO.load("ppo_yaw_controller_era5")

env = YawControlEnv()
obs, _ = env.reset()

for step in range(1000):

    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)

    if step % 100 == 0:
        print(f"Step {step}")
        print("Action:", action)
        print("Yaw Error:", obs[3])
        print("Power:", info["power"])

    if done:
        obs, _ = env.reset()

print("\nTesting Complete ✅")