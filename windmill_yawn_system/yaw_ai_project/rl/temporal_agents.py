"""Sequence-aware DQN agents for wind-turbine yaw control.

This module deliberately leaves :mod:`rl.agents` untouched.  Sequences are
assembled online, are reset for every episode, and are stored atomically with
their next sequence so replay can never cross an episode boundary.
"""
from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.agents import ACTION_SIZE, STATE_SIZE, resolve_device


class SequenceBuilder:
    """Build fixed-length causal state histories without episode leakage.

    At reset the first observed state is left-padded.  There are never future
    observations in a sequence and ``reset`` discards the prior episode.
    """
    def __init__(self, length: int, state_size: int = STATE_SIZE) -> None:
        if length < 1:
            raise ValueError("sequence length must be at least one")
        self.length, self.state_size = int(length), int(state_size)
        self.history: Deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32).reshape(self.state_size)
        self.history.clear()
        for _ in range(self.length):
            self.history.append(value.copy())
        return self.current()

    def append(self, state: np.ndarray) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32).reshape(self.state_size)
        if not self.history:
            return self.reset(value)
        self.history.append(value.copy())
        return self.current()

    def current(self) -> np.ndarray:
        if len(self.history) != self.length:
            raise RuntimeError("SequenceBuilder must be reset before use")
        return np.stack(tuple(self.history), axis=0).astype(np.float32, copy=False)


class SequenceReplayBuffer:
    """Replay buffer whose items already contain causal, same-episode sequences."""
    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=int(capacity))

    def add(self, sequence: np.ndarray, action: int, reward: float, next_sequence: np.ndarray, done: bool) -> None:
        sequence = np.asarray(sequence, dtype=np.float32)
        next_sequence = np.asarray(next_sequence, dtype=np.float32)
        if sequence.ndim != 2 or next_sequence.shape != sequence.shape:
            raise ValueError("sequence and next_sequence must have equal [time, features] shapes")
        self.buffer.append((sequence.copy(), int(action), float(reward), next_sequence.copy(), bool(done)))

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError("not enough sequences to sample")
        sequences, actions, rewards, next_sequences, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.asarray(sequences, dtype=np.float32), np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32), np.asarray(next_sequences, dtype=np.float32),
                np.asarray(dones, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.buffer)


class LSTMQNetwork(nn.Module):
    """LSTM encoder followed by a linear Q-value head."""
    def __init__(self, input_size: int = STATE_SIZE, hidden_size: int = 128, action_size: int = ACTION_SIZE) -> None:
        super().__init__()
        self.input_size, self.hidden_size, self.action_size = input_size, hidden_size, action_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, action_size)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequences)
        return self.head(outputs[:, -1, :])


class AttentionLSTMQNetwork(LSTMQNetwork):
    """LSTM encoder with softmax temporal attention over all hidden states."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, sequences: torch.Tensor, return_attention: bool = False):
        outputs, _ = self.lstm(sequences)
        weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * weights.unsqueeze(-1), dim=1)
        q_values = self.head(context)
        return (q_values, weights) if return_attention else q_values


@dataclass
class TemporalDQNConfig:
    sequence_length: int = 20
    state_size: int = STATE_SIZE
    action_size: int = ACTION_SIZE
    hidden_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    batch_size: int = 128
    train_start_steps: int = 500
    target_update_every: int = 100
    grad_clip_norm: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 60
    device: str = "auto"
    attention: bool = False
    name: str = "lstm_dqn"

    def epsilon_for_episode(self, episode: int) -> float:
        progress = min(1.0, episode / float(self.epsilon_decay_episodes))
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)


class LSTMDQNAgent:
    """DQN with sequence replay, online/target LSTMs and causal TD targets."""
    def __init__(self, config: Optional[TemporalDQNConfig] = None) -> None:
        self.config = config or TemporalDQNConfig()
        self.device = resolve_device(self.config.device)
        network = AttentionLSTMQNetwork if self.config.attention else LSTMQNetwork
        self.online = network(self.config.state_size, self.config.hidden_size, self.config.action_size).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = SequenceReplayBuffer(self.config.buffer_size)
        self.global_step = 0
        self.optimization_steps = 0

    def select_action(self, sequence: np.ndarray, epsilon: float = 0.0, greedy: bool = False) -> int:
        if not greedy and random.random() < epsilon:
            return random.randrange(self.config.action_size)
        with torch.no_grad():
            batch = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(batch).argmax(dim=1).item())

    def attention_weights(self, sequence: np.ndarray) -> Optional[np.ndarray]:
        if not self.config.attention:
            return None
        with torch.no_grad():
            batch = torch.as_tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            _q, weights = self.online(batch, return_attention=True)
        return weights.squeeze(0).cpu().numpy()

    def remember(self, sequence, action, reward, next_sequence, done) -> None:
        self.replay.add(sequence, action, reward, next_sequence, done)

    @torch.no_grad()
    def _target(self, rewards, next_sequences, dones):
        next_q = self.target(next_sequences).max(dim=1, keepdim=True)[0]
        return rewards + self.config.gamma * next_q * (1.0 - dones)

    def optimize(self) -> Optional[Dict[str, float]]:
        if self.global_step < self.config.train_start_steps or len(self.replay) < self.config.batch_size:
            return None
        sequences, actions, rewards, next_sequences, dones = self.replay.sample(self.config.batch_size)
        sequences_t = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_t = torch.as_tensor(next_sequences, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        q = self.online(sequences_t).gather(1, actions_t)
        target = self._target(rewards_t, next_t, dones_t)
        loss = nn.functional.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()
        self.optimization_steps += 1
        return {"loss": float(loss.item()), "grad_norm": float(grad_norm), "avg_q": float(q.detach().mean()), "max_q": float(q.detach().max())}

    def update_target_network(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": self.config.name, "config": asdict(self.config), "model_state_dict": self.online.state_dict(), "target_state_dict": self.target.state_dict(), "optimizer_state_dict": self.optimizer.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "LSTMDQNAgent":
        checkpoint = torch.load(path, map_location=resolve_device(device))
        cfg_data = dict(checkpoint["config"]); cfg_data["device"] = device
        agent = cls(TemporalDQNConfig(**cfg_data))
        agent.online.load_state_dict(checkpoint["model_state_dict"])
        agent.target.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent
