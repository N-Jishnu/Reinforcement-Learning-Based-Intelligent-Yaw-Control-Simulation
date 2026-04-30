import csv
import os
import random
import sys
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from env.environment import YawRLEnvironment


SEED = 42
STATE_SIZE = 5
ACTION_SIZE = 3

DEFAULT_Q_EPISODES = 500
DEFAULT_DQN_EPISODES = 2200
DEFAULT_STEPS_PER_EPISODE = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1500

Q_LEARNING_RATE = 0.05
Q_GAMMA = 0.99
Q_BINS = [24, 12, 24, 24, 24]

DQN_GAMMA = 0.99
DQN_LEARNING_RATE = 3e-4
BUFFER_SIZE = 50000
BATCH_SIZE = 128
TARGET_UPDATE_EVERY = 250
TRAIN_START_STEPS = 2000
GRAD_CLIP_NORM = 5.0

MODEL_PATH = os.path.join("rl", "model.pth")
LOG_PATH = os.path.join("rl", "logs.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, input_size=5, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(ep):
    if ep >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    frac = ep / EPSILON_DECAY_EPISODES
    return EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def discretize_state(state):
    idx = []
    for i, bins in enumerate(Q_BINS):
        val = float(np.clip(state[i], 0.0, 0.999999))
        idx.append(int(val * bins))
    return tuple(idx)


def evaluate_random_policy(env, episodes=20):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, _ = env.step(action)
            total += reward
            done = done or truncated
        rewards.append(total)
    return float(np.mean(rewards))


def train_q_learning(env, q_episodes, steps_per_episode):
    q_table = np.zeros((*Q_BINS, ACTION_SIZE), dtype=np.float32)
    episode_rewards = []

    for episode in range(q_episodes):
        state, _ = env.reset()
        s_idx = discretize_state(state)
        epsilon = epsilon_by_episode(episode)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[s_idx]))

            next_state, reward, done, truncated, _ = env.step(action)
            next_idx = discretize_state(next_state)

            best_next = float(np.max(q_table[next_idx]))
            td_target = reward + Q_GAMMA * best_next * (0.0 if done or truncated else 1.0)
            td_error = td_target - q_table[s_idx + (action,)]
            q_table[s_idx + (action,)] += Q_LEARNING_RATE * td_error

            s_idx = next_idx
            total_reward += reward
            if done or truncated:
                break

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_50 = float(np.mean(episode_rewards[-50:]))
            print(f"[Q] Episode {episode + 1}/{q_episodes}, Avg(50): {avg_50:.3f}, Eps: {epsilon:.3f}")

    return q_table, episode_rewards


def select_action_dqn(policy_net, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(policy_net, target_net, optimizer, replay_buffer, global_step):
    if global_step <= TRAIN_START_STEPS:
        return None
    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    current_q = policy_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + DQN_GAMMA * next_q * (1.0 - dones_t)

    loss = nn.MSELoss()(current_q, target_q)

    optimizer.zero_grad()
    if global_step > TRAIN_START_STEPS:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

    return float(loss.item())


def train_dqn(env, dqn_episodes, steps_per_episode):
    policy_net = QNetwork(input_size=STATE_SIZE, output_size=ACTION_SIZE)
    target_net = QNetwork(input_size=STATE_SIZE, output_size=ACTION_SIZE)
    policy_net.to(DEVICE)
    target_net.to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=DQN_LEARNING_RATE)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    episode_rewards = []
    losses = []
    global_step = 0

    for episode in range(dqn_episodes):
        state, _ = env.reset()
        epsilon = epsilon_by_episode(episode)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            action = select_action_dqn(policy_net, state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            terminal = done or truncated

            replay_buffer.add(state, action, reward, next_state, terminal)
            state = next_state
            total_reward += reward

            loss = optimize_dqn(policy_net, target_net, optimizer, replay_buffer, global_step)
            if loss is not None:
                losses.append(loss)

            global_step += 1
            if global_step % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if terminal:
                break

        episode_rewards.append(total_reward)
        if (episode + 1) % 50 == 0:
            avg_50 = float(np.mean(episode_rewards[-50:]))
            loss_str = f", Loss: {np.mean(losses[-200:]):.4f}" if losses else ""
            print(f"[DQN] Episode {episode + 1}/{dqn_episodes}, Avg(50): {avg_50:.3f}, Eps: {epsilon:.3f}{loss_str}")

    return policy_net, episode_rewards


def write_logs(log_rows):
    os.makedirs("rl", exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "episode",
                "episode_reward",
                "average_reward_50",
                "epsilon",
            ],
        )
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Q-learning baseline and DQN yaw controller")
    parser.add_argument("--q-episodes", type=int, default=DEFAULT_Q_EPISODES)
    parser.add_argument("--dqn-episodes", type=int, default=DEFAULT_DQN_EPISODES)
    parser.add_argument("--steps-per-episode", type=int, default=DEFAULT_STEPS_PER_EPISODE)
    parser.add_argument("--eval-random-episodes", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    os.makedirs("rl", exist_ok=True)

    env = YawRLEnvironment()
    print(f"Using device: {DEVICE}")

    random_baseline = evaluate_random_policy(env, episodes=args.eval_random_episodes)
    print(f"Random policy average reward: {random_baseline:.3f}")

    q_table, q_rewards = train_q_learning(env, args.q_episodes, args.steps_per_episode)
    dqn_model, dqn_rewards = train_dqn(env, args.dqn_episodes, args.steps_per_episode)

    torch.save(
        {
            "model_state_dict": dqn_model.state_dict(),
            "state_size": STATE_SIZE,
            "action_size": ACTION_SIZE,
            "hidden_layers": [64, 64],
        },
        MODEL_PATH,
    )

    logs = []
    for i, reward in enumerate(q_rewards):
        logs.append(
            {
                "phase": "q_learning",
                "episode": i + 1,
                "episode_reward": float(reward),
                "average_reward_50": float(np.mean(q_rewards[max(0, i - 49): i + 1])),
                "epsilon": float(epsilon_by_episode(i)),
            }
        )
    for i, reward in enumerate(dqn_rewards):
        logs.append(
            {
                "phase": "dqn",
                "episode": i + 1,
                "episode_reward": float(reward),
                "average_reward_50": float(np.mean(dqn_rewards[max(0, i - 49): i + 1])),
                "epsilon": float(epsilon_by_episode(i)),
            }
        )
    write_logs(logs)

    q_last = float(np.mean(q_rewards[-50:]))
    dqn_last = float(np.mean(dqn_rewards[-50:]))
    print(f"Q-learning Avg(Last 50): {q_last:.3f}")
    print(f"DQN Avg(Last 50): {dqn_last:.3f}")
    print(f"Improvement over random (DQN): {dqn_last - random_baseline:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved logs: {LOG_PATH}")


if __name__ == "__main__":
    main()
