"""Training/evaluation loops for causal sequence DQN agents."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from rl.temporal_agents import LSTMDQNAgent, SequenceBuilder
from rl.trainer import evaluate_policy

def train_temporal_agent(env, agent: LSTMDQNAgent, episodes: int, steps_per_episode: int, output_dir: str | Path, prefix: str, seed: int) -> Dict[str, float]:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    step_rows: List[Dict[str, float]] = []; episode_rows: List[Dict[str, float]] = []; rewards: List[float] = []
    start = time.perf_counter()
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        builder = SequenceBuilder(agent.config.sequence_length, agent.config.state_size)
        sequence = builder.reset(state); epsilon = agent.config.epsilon_for_episode(ep)
        total = 0.0; updates = []
        episode_length = 0
        for _step in range(steps_per_episode):
            action = agent.select_action(sequence, epsilon=epsilon)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = bool(terminated or truncated)
            next_sequence = builder.append(next_state)
            agent.remember(sequence, action, reward, next_sequence, done)
            update = agent.optimize()
            if update:
                updates.append(update)
                step_rows.append({"episode": ep + 1, "global_step": agent.global_step, "epsilon": epsilon, **update})
            agent.global_step += 1
            if agent.global_step % agent.config.target_update_every == 0: agent.update_target_network()
            sequence = next_sequence; total += float(reward); episode_length += 1
            if done: break
        rewards.append(total)
        episode_rows.append({"episode": ep + 1, "episode_reward": total, "moving_avg_50": float(np.mean(rewards[-50:])), "epsilon": epsilon, "episode_length": episode_length, "mean_loss": float(np.mean([x['loss'] for x in updates])) if updates else float('nan'), "mean_grad_norm": float(np.mean([x['grad_norm'] for x in updates])) if updates else float('nan')})
    steps = pd.DataFrame(step_rows); episodes_frame = pd.DataFrame(episode_rows)
    steps.to_csv(output_dir / f"{prefix}_metrics.csv", index=False); episodes_frame.to_csv(output_dir / f"{prefix}_episodes.csv", index=False)
    return {"wall_time_s": time.perf_counter() - start, "final_avg_reward_50": float(np.mean(rewards[-50:])), "mean_loss": float(steps.loss.mean()) if not steps.empty else float('nan'), "mean_grad_norm": float(steps.grad_norm.mean()) if not steps.empty else float('nan')}

def evaluate_temporal_agent(env, agent: LSTMDQNAgent, episodes: int, seed_base: int, attention_path: str | Path | None = None):
    # A fresh builder is created for each episode: evaluation also cannot leak history.
    rows = []; attention_rows = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed_base + episode); builder = SequenceBuilder(agent.config.sequence_length); sequence = builder.reset(state); done = False
        rewards=[]; power=[]; physical=[]; mis=[]; moves=[]; times=[]
        from simulation.power_model import compute_power
        while not done:
            t0=time.perf_counter(); action=agent.select_action(sequence, greedy=True); times.append((time.perf_counter()-t0)*1000)
            weights=agent.attention_weights(sequence)
            next_state, reward, terminated, truncated, info=env.step(action); done=bool(terminated or truncated)
            if weights is not None: attention_rows.append({"episode":episode+1,"step":len(rewards), **{f"lag_{i}": float(w) for i,w in enumerate(weights)}})
            rewards.append(float(reward)); mis.append(float(info['misalignment'])); moves.append(float(info['yaw_change']))
            power.append(float(compute_power(float(info['wind_speed']), float(info['misalignment']), mode='normalized')))
            physical.append(float(compute_power(float(info['wind_speed']), float(info['misalignment']), mode='physical')))
            sequence=builder.append(next_state)
        rows.append({"episode_reward":float(np.sum(rewards)),"average_power_normalized":float(np.mean(power)),"average_power_physical":float(np.mean(physical)),"mean_abs_misalignment":float(np.mean(np.abs(mis))),"yaw_movement_frequency":float(np.mean(np.abs(moves)>1e-6)),"control_stability":float(np.std(moves)),"inference_time_ms":float(np.mean(times))})
    if attention_path is not None and attention_rows: pd.DataFrame(attention_rows).to_csv(attention_path, index=False)
    return rows
