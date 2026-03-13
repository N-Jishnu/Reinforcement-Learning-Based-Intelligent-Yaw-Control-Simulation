import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from simulation.yaw_env import YawControlEnv

# ----------------------------------
# Configuration
# ----------------------------------

TOTAL_TIMESTEPS = 700_000
MODEL_NAME = "ppo_yaw_controller_era5"
LOG_DIR = "./logs/"
CHECKPOINT_DIR = "./checkpoints/"
SEED = 42

# ----------------------------------
# Main Training Function
# ----------------------------------

def main():

    print("🚀 Training PPO on ERA5 Real Wind")

    # Ensure folders exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Reproducibility
    np.random.seed(SEED)

    # ⭐ Environment wrapper (required by SB3)
    env = DummyVecEnv([lambda: YawControlEnv()])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",              # Correct for MLP policies
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR
    )

    # ⭐ Save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_yaw"
    )

    print(f"⏳ Training for {TOTAL_TIMESTEPS:,} timesteps...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save(MODEL_NAME)

    print("✅ ERA5 PPO Training Complete")
    print(f"💾 Model saved as: {MODEL_NAME}")


# ----------------------------------
# Entry Point
# ----------------------------------

if __name__ == "__main__":
    main()