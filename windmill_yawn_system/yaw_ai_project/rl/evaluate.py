import argparse
import os
import sys
import random

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import YawRLEnvironment
from rl.train import QNetwork


MODEL_PATH = os.path.join("rl", "model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_random(env, episodes):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            done = done or truncated
        rewards.append(total_reward)
    return np.array(rewards, dtype=float)


def evaluate_dqn(env, model, episodes):
    rewards = []
    model.eval()
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_t)
                action = int(torch.argmax(q_values, dim=1).item())
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            done = done or truncated
        rewards.append(total_reward)
    return np.array(rewards, dtype=float)


def load_model(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_size = int(ckpt.get("state_size", 5))
    action_size = int(ckpt.get("action_size", 3))
    model = QNetwork(input_size=state_size, output_size=action_size).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN vs random policy")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    set_seed(args.seed)
    env = YawRLEnvironment()
    model = load_model(args.model_path)

    random_rewards = evaluate_random(env, args.episodes)
    dqn_rewards = evaluate_dqn(env, model, args.episodes)

    random_mean = float(np.mean(random_rewards))
    random_std = float(np.std(random_rewards))
    dqn_mean = float(np.mean(dqn_rewards))
    dqn_std = float(np.std(dqn_rewards))
    improvement = dqn_mean - random_mean

    print(f"Using device: {DEVICE}")
    print(f"Episodes: {args.episodes}")
    print(f"Random mean +/- std: {random_mean:.3f} +/- {random_std:.3f}")
    print(f"DQN mean +/- std:    {dqn_mean:.3f} +/- {dqn_std:.3f}")
    print(f"Improvement:         {improvement:.3f}")

    if improvement > 0:
        print("Result: PASS (DQN better than random policy)")
    else:
        print("Result: NOT YET (DQN not better than random policy)")


if __name__ == "__main__":
    main()
