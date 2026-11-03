"""Shared neural-network training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from rl.agents import DQNAgent
from rl.metrics import MetricLogger


def train_agent(
    env,
    agent: DQNAgent,
    episodes: int,
    steps_per_episode: int,
    logger: MetricLogger,
    log_interval: int = 100,
    print_interval: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Train a DQN/Double DQN agent and log real metrics.

    Step-level metrics are written every ``log_interval`` optimization steps.
    Episode-level metrics are written after every episode.  No synthetic loss
    or gradient values are used: the numbers come directly from the optimizer
    step and from ``clip_grad_norm_``'s returned norm.
    """

    start_time = time.perf_counter()
    episode_rewards: List[float] = []
    total_optimization_steps = 0
    all_losses: List[float] = []
    all_grad_norms: List[float] = []

    for episode in range(episodes):
        episode_start = time.perf_counter()
        episode_seed = None if seed is None else int(seed + episode)
        state, _ = env.reset(seed=episode_seed)
        epsilon = agent.epsilon_for_episode(episode)
        total_reward = 0.0
        episode_length = 0

        episode_losses: List[float] = []
        episode_grads: List[float] = []
        episode_qs: List[float] = []
        episode_max_qs: List[float] = []
        optimization_steps_before = agent.optimization_steps

        for _ in range(steps_per_episode):
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)

            if agent.config.use_replay:
                agent.remember(state, action, float(reward), next_state, done)
                update = agent.optimize()
            else:
                update = agent.update_single(state, action, float(reward), next_state, done)

            if update is not None:
                episode_losses.append(update["loss"])
                episode_grads.append(update["grad_norm"])
                episode_qs.append(update["avg_q"])
                episode_max_qs.append(update["max_q"])
                all_losses.append(update["loss"])
                all_grad_norms.append(update["grad_norm"])

                if log_interval > 0 and agent.global_step % log_interval == 0:
                    logger.log_step(
                        episode=episode + 1,
                        global_step=agent.global_step,
                        loss=update["loss"],
                        grad_norm=update["grad_norm"],
                        avg_q=update["avg_q"],
                        max_q=update["max_q"],
                        epsilon=epsilon,
                        learning_rate=agent.config.learning_rate,
                    )

            agent.global_step += 1
            if agent.target is not None and agent.config.target_update_every > 0:
                if agent.global_step % agent.config.target_update_every == 0:
                    agent.update_target_network()

            state = next_state
            total_reward += float(reward)
            episode_length += 1
            if done:
                break

        episode_time = time.perf_counter() - episode_start
        total_time = time.perf_counter() - start_time
        optimization_steps = agent.optimization_steps - optimization_steps_before
        total_optimization_steps += optimization_steps
        episode_rewards.append(total_reward)

        # A zero before the first optimizer update is not a measurement. Keep
        # unavailable aggregates explicit so warm-up rows cannot be mistaken
        # for genuine zero loss, Q-value, or gradient observations.
        unavailable = float("nan")
        logger.log_episode(
            episode=episode + 1,
            episode_reward=total_reward,
            moving_avg_50=float(np.mean(episode_rewards[-50:])),
            epsilon=epsilon,
            episode_length=episode_length,
            episode_time_s=episode_time,
            total_time_s=total_time,
            optimization_steps=optimization_steps,
            mean_loss=float(np.mean(episode_losses)) if episode_losses else unavailable,
            mean_grad_norm=float(np.mean(episode_grads)) if episode_grads else unavailable,
            mean_q=float(np.mean(episode_qs)) if episode_qs else unavailable,
            max_q=float(np.max(episode_max_qs)) if episode_max_qs else unavailable,
            learning_rate=agent.config.learning_rate,
        )
        logger.flush()

        if print_interval and (episode + 1) % print_interval == 0:
            print(
                f"[{agent.config.name.upper()}] Episode {episode + 1}/{episodes}, "
                f"Avg(50): {np.mean(episode_rewards[-50:]):.3f}, "
                f"Eps: {epsilon:.3f}, "
                f"Loss: {np.mean(episode_losses) if episode_losses else 0.0:.4f}"
            )

    wall_time = time.perf_counter() - start_time
    final_avg = float(np.mean(episode_rewards[-50:])) if episode_rewards else float("nan")
    summary: Dict[str, object] = {
        "algorithm": agent.config.name,
        "run_name": logger.prefix,
        "seed": seed,
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "total_steps": agent.global_step,
        "total_optimization_steps": total_optimization_steps,
        "wall_time_s": wall_time,
        "final_episode_reward": episode_rewards[-1] if episode_rewards else float("nan"),
        "final_avg_reward_50": final_avg,
        "best_avg_reward_50": _best_moving_average(episode_rewards, window=50),
        "mean_loss": float(np.mean(all_losses)) if all_losses else float("nan"),
        "mean_grad_norm": float(np.mean(all_grad_norms)) if all_grad_norms else float("nan"),
        "total_trainable_parameters": sum(p.numel() for p in agent.online.parameters() if p.requires_grad),
        "device": str(agent.device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "notes": "reduced experimental budget" if episodes < 100 else "",
    }
    logger.log_summary(summary)
    # Returned for the legacy ``rl/logs.csv`` writer; not written to the
    # summary CSV because it is a per-episode list.
    summary["episode_rewards"] = episode_rewards
    return summary


def _best_moving_average(rewards: List[float], window: int = 50) -> float:
    if not rewards:
        return float("nan")
    best = float("-inf")
    for i in range(len(rewards)):
        value = float(np.mean(rewards[max(0, i - window + 1): i + 1]))
        best = max(best, value)
    return best


def _episode_metrics(
    rewards: List[float],
    normalized_power: List[float],
    physical_power: List[float],
    misalignments: List[float],
    yaw_changes: List[float],
    inference_times_ms: List[float],
    dt: float = 1.0,
) -> Dict[str, float]:
    yaw_changes_arr = np.asarray(yaw_changes, dtype=float)
    misalignment_arr = np.asarray(misalignments, dtype=float)
    non_zero_moves = yaw_changes_arr[np.abs(yaw_changes_arr) > 1e-9]
    reversals = 0
    for i in range(1, len(non_zero_moves)):
        if np.sign(non_zero_moves[i]) != np.sign(non_zero_moves[i - 1]):
            reversals += 1
    return {
        "episode_reward": float(np.sum(rewards)),
        "average_power_normalized": float(np.mean(normalized_power)) if normalized_power else float("nan"),
        "average_power_physical": float(np.mean(physical_power)) if physical_power else float("nan"),
        "total_energy_normalized": float(np.sum(normalized_power) * dt),
        "total_energy_physical": float(np.sum(physical_power) * dt),
        "mean_abs_misalignment": float(np.mean(np.abs(misalignment_arr))) if len(misalignment_arr) else float("nan"),
        "yaw_movement_frequency": float(np.mean(np.abs(yaw_changes_arr) > 1e-6)) if len(yaw_changes_arr) else 0.0,
        "control_stability": float(np.std(yaw_changes_arr)) if len(yaw_changes_arr) else 0.0,
        "cumulative_yaw_movement": float(np.sum(np.abs(yaw_changes_arr))),
        "yaw_reversals": int(reversals),
        "time_within_10_deg": float(np.mean(np.abs(misalignment_arr) <= 10.0)) if len(misalignment_arr) else 0.0,
        "inference_time_ms": float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
    }


def evaluate_policy(
    env,
    action_fn: Callable[[np.ndarray], int],
    episodes: int,
    seed_base: int = 10_000,
    dt: float = 1.0,
    measure_inference: bool = False,
) -> List[Dict[str, float]]:
    """Evaluate a policy on fixed, independently seeded wind episodes.

    The returned metrics include both power modes and the additional mechanics
    metrics requested for the upgraded evaluation (time within 10 degrees,
    cumulative movement and reversals).
    """

    from simulation.power_model import compute_power

    results: List[Dict[str, float]] = []
    for episode in range(episodes):
        state, _ = env.reset(seed=int(seed_base + episode))
        rewards: List[float] = []
        normalized: List[float] = []
        physical: List[float] = []
        misalignments: List[float] = []
        yaw_changes: List[float] = []
        inference_times: List[float] = []
        done = False
        while not done:
            if measure_inference:
                t0 = time.perf_counter()
                action = action_fn(state)
                inference_times.append((time.perf_counter() - t0) * 1000.0)
            else:
                action = action_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rewards.append(float(reward))
            misalignment = float(info["misalignment"])
            wind_speed = float(info["wind_speed"])
            yaw_changes.append(float(info["yaw_change"]))
            misalignments.append(misalignment)
            normalized.append(float(compute_power(wind_speed, misalignment, mode="normalized")))
            physical.append(float(compute_power(wind_speed, misalignment, mode="physical")))
        results.append(
            _episode_metrics(
                rewards,
                normalized,
                physical,
                misalignments,
                yaw_changes,
                inference_times,
                dt=dt,
            )
        )
    return results


def evaluate_agent(
    env,
    agent: DQNAgent,
    episodes: int,
    seed_base: int = 10_000,
) -> List[Dict[str, float]]:
    return evaluate_policy(
        env,
        action_fn=lambda state: agent.select_action(state, greedy=True),
        episodes=episodes,
        seed_base=seed_base,
        measure_inference=True,
    )


def evaluate_random(
    env,
    episodes: int,
    seed_base: int = 10_000,
    seed: int = 42,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    return evaluate_policy(
        env,
        action_fn=lambda _state: int(rng.integers(0, env.action_space.n)),
        episodes=episodes,
        seed_base=seed_base,
    )
