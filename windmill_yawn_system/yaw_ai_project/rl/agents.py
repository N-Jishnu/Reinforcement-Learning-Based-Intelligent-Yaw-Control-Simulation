"""Neural-network agents and replay memory for wind-turbine yaw control.

This module contains the shared PyTorch implementation used by the original
DQN pipeline and by the new Double DQN and ablation experiments.  The network
architecture is intentionally unchanged from the original project:
``5 -> 128 -> 128 -> 3`` with ReLU activations.
"""

from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

STATE_SIZE = 5
ACTION_SIZE = 3
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (128, 128)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` asks PyTorch for deterministic CuDNN behaviour.
    Some CUDA operations may still be non-deterministic; this is documented in
    the project documentation.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` into a ``torch.device``."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infer_hidden_sizes(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, ...]:
    """Infer hidden layer widths from a ``QNetwork`` state dict.

    This makes checkpoint loading robust against older checkpoints whose
    metadata (``hidden_layers``) was incorrect.  The last ``net.N.weight``
    entry is the output layer and is therefore excluded.
    """

    weight_shapes = []
    for key, value in state_dict.items():
        if key.startswith("net.") and key.endswith(".weight") and value.ndim == 2:
            try:
                index = int(key.split(".")[1])
            except (IndexError, ValueError):
                continue
            weight_shapes.append((index, int(value.shape[0])))
    if len(weight_shapes) < 2:
        return DEFAULT_HIDDEN_SIZES
    weight_shapes.sort(key=lambda item: item[0])
    return tuple(size for _index, size in weight_shapes[:-1])


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron that maps a 5-D state to three Q-values.

    Layers
    ------
    Linear(5, 128) -> ReLU
    Linear(128, 128) -> ReLU
    Linear(128, 3)

    The default initialisation is PyTorch's standard ``Linear`` initialisation
    (Kaiming-uniform weights and uniform biases).  The original project did not
    override it, so neither does this implementation.
    """

    def __init__(
        self,
        input_size: int = STATE_SIZE,
        output_size: int = ACTION_SIZE,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        layers: List[nn.Module] = []
        previous = self.input_size
        for hidden in self.hidden_sizes:
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            previous = hidden
        layers.append(nn.Linear(previous, self.output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -- introspection -----------------------------------------------------

    def layer_summary(self) -> List[Dict[str, object]]:
        """Return one row per learnable layer for reports and dashboards."""

        rows: List[Dict[str, object]] = []
        linear_index = 0
        for module in self.net:
            if isinstance(module, nn.Linear):
                rows.append(
                    {
                        "layer": f"Linear_{linear_index}",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "activation": "ReLU" if linear_index < len(self.hidden_sizes) else "None (linear output)",
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                )
                linear_index += 1
        return rows

    def parameter_summary(self) -> Dict[str, object]:
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "input_shape": (self.input_size,),
            "output_shape": (self.output_size,),
            "hidden_sizes": list(self.hidden_sizes),
            "num_linear_layers": len(self.hidden_sizes) + 1,
            "total_trainable_parameters": int(total),
            "layers": self.layer_summary(),
        }

    def describe(self) -> str:
        """Human-readable architecture summary."""

        summary = self.parameter_summary()
        lines = [
            "QNetwork architecture",
            "---------------------",
            f"Input shape:  {summary['input_shape']}",
            f"Hidden sizes: {summary['hidden_sizes']}",
            f"Output shape: {summary['output_shape']}",
            f"Linear layers: {summary['num_linear_layers']}",
            f"Trainable parameters: {summary['total_trainable_parameters']:,}",
            "",
            "Per-layer parameters:",
        ]
        for row in summary["layers"]:
            lines.append(
                "  {layer:<10} {in_features:>4} -> {out_features:<4} "
                "activation={activation:<20} params={parameters:,}".format(**row)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Replay memory
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity experience replay buffer.

    Transitions are stored as ``(state, action, reward, next_state, done)``.
    Sampling returns NumPy arrays so the trainer can move them to the correct
    device in one operation.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {len(self.buffer)}"
            )
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()


# ---------------------------------------------------------------------------
# Agent configuration and DQN
# ---------------------------------------------------------------------------

@dataclass
class DQNConfig:
    """Hyperparameters for DQN and Double DQN.

    Defaults match the original ``rl/train.py`` so existing behaviour is
    preserved.  Experimental scripts may override episodes/steps, but the
    learning rate, discount, replay capacity and batch size are kept aligned
    with the original implementation wherever possible.
    """

    state_size: int = STATE_SIZE
    action_size: int = ACTION_SIZE
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    batch_size: int = 256
    target_update_every: int = 250
    train_start_steps: int = 2_000
    grad_clip_norm: float = 5.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_500
    double_dqn: bool = False
    use_replay: bool = True
    use_target_network: bool = True
    device: str = "auto"
    seed: int = 42
    name: str = "dqn"

    def epsilon_for_episode(self, episode: int) -> float:
        if episode >= self.epsilon_decay_steps:
            return self.epsilon_end
        fraction = episode / float(self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


class DQNAgent:
    """Standard DQN agent with experience replay and a target network.

    The agent is deliberately small and explicit so that every neural-network
    operation (forward pass, loss, backward pass, gradient clipping, target
    synchronisation) is visible to students and reviewers.
    """

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.device = resolve_device(self.config.device)

        self.online = QNetwork(
            input_size=self.config.state_size,
            output_size=self.config.action_size,
            hidden_sizes=self.config.hidden_sizes,
        ).to(self.device)
        self.target: Optional[QNetwork] = None
        if self.config.use_target_network:
            self.target = copy.deepcopy(self.online).to(self.device)
            self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.replay = ReplayBuffer(self.config.buffer_size) if self.config.use_replay else None

        self.global_step = 0
        self.optimization_steps = 0
        self.last_update: Optional[Dict[str, float]] = None

    # -- exploration / action selection -----------------------------------

    def epsilon_for_episode(self, episode: int) -> float:
        return self.config.epsilon_for_episode(episode)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0, greedy: bool = False) -> int:
        """Epsilon-greedy action selection.

        When ``greedy`` is true (evaluation), exploration is disabled.
        """

        if not greedy and random.random() < epsilon:
            return random.randint(0, self.config.action_size - 1)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Return Q-values for a batch of states (used by dashboards/tests)."""

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.online(states_t).cpu().numpy()

    # -- memory ------------------------------------------------------------

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self.replay is not None:
            self.replay.add(state, action, reward, next_state, done)

    # -- learning ----------------------------------------------------------

    @torch.no_grad()
    def compute_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the bootstrapped target.

        Standard DQN:
            y = r + gamma * max_a Q_target(s', a) * (1 - done)

        Double DQN:
            a* = argmax_a Q_online(s', a)
            y  = r + gamma * Q_target(s', a*) * (1 - done)
        """

        if self.target is not None:
            target_net = self.target
        else:
            # Ablation: no separate target network.
            target_net = self.online

        next_q_target = target_net(next_states)
        if self.config.double_dqn and self.target is not None:
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = next_q_target.gather(1, next_actions)
        else:
            next_q = next_q_target.max(dim=1, keepdim=True)[0]

        return rewards + self.config.gamma * next_q * (1.0 - dones)

    def _optimize_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Dict[str, float]:
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.online(states_t).gather(1, actions_t)
        target_q = self.compute_target(rewards_t, next_states_t, dones_t)

        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        self.optimization_steps += 1

        result = {
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "avg_q": float(current_q.detach().mean().item()),
            "max_q": float(current_q.detach().max().item()),
        }
        self.last_update = result
        return result

    def optimize(self) -> Optional[Dict[str, float]]:
        """One gradient update from a replay mini-batch."""

        if self.replay is None:
            return None
        if self.global_step <= self.config.train_start_steps:
            return None
        if len(self.replay) < self.config.batch_size:
            return None
        batch = self.replay.sample(self.config.batch_size)
        return self._optimize_batch(*batch)

    def update_single(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Dict[str, float]]:
        """Online update for the no-replay ablation.

        A batch of one is used because retaining past transitions would defeat
        the purpose of the ablation.  This produces high-variance gradients and
        is expected to be less stable than replay-based training.
        """

        if self.global_step <= self.config.train_start_steps:
            return None
        return self._optimize_batch(
            np.asarray([state], dtype=np.float32),
            np.asarray([action], dtype=np.int64),
            np.asarray([reward], dtype=np.float32),
            np.asarray([next_state], dtype=np.float32),
            np.asarray([done], dtype=np.float32),
        )

    def update_target_network(self) -> None:
        if self.target is not None:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()

    # -- persistence -------------------------------------------------------

    def checkpoint(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model_state_dict": self.online.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_size": self.config.state_size,
            "action_size": self.config.action_size,
            "hidden_layers": list(self.config.hidden_sizes),
            "algorithm": "double_dqn" if self.config.double_dqn else "dqn",
            "config": asdict(self.config),
        }
        if self.target is not None:
            payload["target_state_dict"] = self.target.state_dict()
        return payload

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), target)

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device)
        self.online.load_state_dict(checkpoint["model_state_dict"])
        if self.target is not None and "target_state_dict" in checkpoint:
            self.target.load_state_dict(checkpoint["target_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except (ValueError, KeyError):
                # Checkpoint may come from a different optimizer/configuration.
                pass

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        config: Optional[DQNConfig] = None,
        device: str = "auto",
    ) -> "DQNAgent":
        checkpoint = torch.load(Path(path), map_location=resolve_device(device))
        cfg = config
        if cfg is None:
            state_dict = checkpoint.get("model_state_dict", {})
            # Prefer the real tensor shapes over metadata, because some legacy
            # checkpoints recorded ``hidden_layers`` incorrectly.
            hidden_sizes = infer_hidden_sizes(state_dict)
            if not hidden_sizes:
                hidden_sizes = tuple(checkpoint.get("hidden_layers", DEFAULT_HIDDEN_SIZES))
            cfg = DQNConfig(
                state_size=int(checkpoint.get("state_size", STATE_SIZE)),
                action_size=int(checkpoint.get("action_size", ACTION_SIZE)),
                hidden_sizes=hidden_sizes,
                device=device,
                double_dqn=str(checkpoint.get("algorithm", "dqn")) == "double_dqn",
            )
        # ``from_checkpoint`` lives on ``DQNAgent`` so that legacy code can
        # call it directly; dispatch to ``DoubleDQNAgent`` when the checkpoint
        # was produced by the Double DQN implementation.
        if cfg.double_dqn and cls is DQNAgent:
            cfg.name = "double_dqn"
            agent = DoubleDQNAgent(cfg)
        else:
            agent = cls(cfg)
        agent.load(path, load_optimizer=False)
        return agent

    # -- reporting ---------------------------------------------------------

    def architecture_summary(self) -> Dict[str, object]:
        summary = self.online.parameter_summary()
        summary.update(
            {
                "algorithm": "double_dqn" if self.config.double_dqn else "dqn",
                "device": str(self.device),
                "optimizer": "Adam",
                "learning_rate": self.config.learning_rate,
                "loss": "MSE",
                "gamma": self.config.gamma,
                "batch_size": self.config.batch_size,
                "replay_capacity": self.config.buffer_size if self.config.use_replay else 0,
                "target_network": self.config.use_target_network,
                "double_dqn": self.config.double_dqn,
            }
        )
        return summary


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent.

    The online network selects the next action; the target network evaluates
    that action.  This reduces the over-estimation bias of standard DQN while
    retaining experience replay and a target network.
    """

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        if config is None:
            config = DQNConfig(double_dqn=True, name="double_dqn")
        else:
            config = copy.deepcopy(config)
            config.double_dqn = True
            config.name = "double_dqn"
        super().__init__(config)


# ---------------------------------------------------------------------------
# Q-learning helper (kept for comparison experiments)
# ---------------------------------------------------------------------------

Q_BINS = [24, 12, 24, 24, 24]


def discretize_state(state: np.ndarray, bins: Sequence[int] = Q_BINS) -> Tuple[int, ...]:
    indices = []
    for i, n_bins in enumerate(bins):
        value = float(np.clip(state[i], 0.0, 0.999999))
        indices.append(int(value * n_bins))
    return tuple(indices)


def train_tabular_q_learning(
    env,
    episodes: int,
    steps_per_episode: int,
    learning_rate: float = 0.05,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 500,
    bins: Sequence[int] = Q_BINS,
    seed: Optional[int] = None,
    log_interval: int = 50,
) -> Tuple[np.ndarray, List[float], List[Dict[str, float]]]:
    """Train the original tabular Q-learning baseline.

    Returns the Q-table, episode rewards and per-episode metric dictionaries.
    The function mirrors the update rule in the original ``rl/train.py``.
    """

    q_table = np.zeros((*bins, ACTION_SIZE), dtype=np.float32)
    rewards: List[float] = []
    history: List[Dict[str, float]] = []

    for episode in range(episodes):
        if seed is not None:
            state, _ = env.reset(seed=seed + episode)
        else:
            state, _ = env.reset()
        s_idx = discretize_state(state, bins)
        if episode >= epsilon_decay_episodes:
            epsilon = epsilon_end
        else:
            epsilon = epsilon_start + (episode / float(epsilon_decay_episodes)) * (
                epsilon_end - epsilon_start
            )
        total_reward = 0.0
        length = 0
        for _ in range(steps_per_episode):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[s_idx]))
            next_state, reward, done, truncated, _ = env.step(action)
            next_idx = discretize_state(next_state, bins)
            best_next = float(np.max(q_table[next_idx]))
            td_target = reward + gamma * best_next * (0.0 if done or truncated else 1.0)
            q_table[s_idx + (action,)] += learning_rate * (td_target - q_table[s_idx + (action,)])
            s_idx = next_idx
            total_reward += float(reward)
            length += 1
            if done or truncated:
                break
        rewards.append(total_reward)
        history.append(
            {
                "episode": episode + 1,
                "episode_reward": total_reward,
                "moving_avg_50": float(np.mean(rewards[-50:])),
                "epsilon": epsilon,
                "episode_length": length,
            }
        )
        if log_interval and (episode + 1) % log_interval == 0:
            print(
                f"[Q] Episode {episode + 1}/{episodes}, "
                f"Avg(50): {np.mean(rewards[-50:]):.3f}, Eps: {epsilon:.3f}"
            )
    return q_table, rewards, history
