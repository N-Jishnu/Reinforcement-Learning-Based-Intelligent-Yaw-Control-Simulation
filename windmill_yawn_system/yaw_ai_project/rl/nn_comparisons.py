"""Controlled optimizer and activation experiments for the unchanged DQN task."""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.environment import YawRLEnvironment
from rl.agents import DQNAgent, DQNConfig, resolve_device, set_global_seed
from rl.experiments import PROFILES, summarise_runs
from rl.metrics import MetricLogger
from rl.trainer import evaluate_policy, train_agent

ACTIVATIONS = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}

class ConfigurableQNetwork(nn.Module):
    """Same 5->128->128->3 topology as DQN; only hidden activation varies."""
    def __init__(self, activation: str = "relu", input_size: int = 5, output_size: int = 3, hidden=(128,128)):
        super().__init__(); self.activation_name = activation
        activation_cls = ACTIVATIONS[activation]
        self.net = nn.Sequential(nn.Linear(input_size, hidden[0]), activation_cls(), nn.Linear(hidden[0], hidden[1]), activation_cls(), nn.Linear(hidden[1], output_size))
    def forward(self, x): return self.net(x)

class ComparisonDQNAgent(DQNAgent):
    """DQN mechanics preserved while optimizer/hidden activation are controlled."""
    def __init__(self, config: DQNConfig, optimizer_name: str = "adam", activation: str = "relu"):
        self.config = config; self.device = resolve_device(config.device); self.optimizer_name=optimizer_name; self.activation_name=activation
        self.online = ConfigurableQNetwork(activation, config.state_size, config.action_size, config.hidden_sizes).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device); self.target.eval()
        optimizers = {"adam": lambda: optim.Adam(self.online.parameters(), lr=config.learning_rate), "rmsprop": lambda: optim.RMSprop(self.online.parameters(), lr=config.learning_rate), "sgd_momentum": lambda: optim.SGD(self.online.parameters(), lr=config.learning_rate, momentum=0.9)}
        self.optimizer = optimizers[optimizer_name]()
        from rl.agents import ReplayBuffer
        self.replay = ReplayBuffer(config.buffer_size); self.global_step=0; self.optimization_steps=0; self.last_update=None

def _convergence_episode(episodes: pd.DataFrame) -> float:
    if episodes.empty or not np.isfinite(episodes.moving_avg_50).any(): return float("nan")
    threshold = float(episodes.moving_avg_50.max()) * 0.9
    hit = episodes.loc[episodes.moving_avg_50 >= threshold, "episode"]
    return float(hit.iloc[0]) if not hit.empty else float("nan")

def run_comparison(kind: str, seeds: List[int], profile_name: str, output_dir: Path, device: str, eval_episodes: int) -> None:
    if kind not in ("optimizer", "activation"): raise ValueError(kind)
    variants = ["adam","rmsprop","sgd_momentum"] if kind == "optimizer" else list(ACTIVATIONS)
    profile=dict(PROFILES[profile_name]); rows=[]; output_dir.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        for seed in seeds:
            set_global_seed(seed)
            config=DQNConfig(batch_size=profile["batch_size"], buffer_size=profile["buffer_size"], train_start_steps=profile["train_start_steps"], target_update_every=profile["target_update_every"], epsilon_decay_steps=profile["epsilon_decay_steps"], device=device, seed=seed, name=f"{kind}_{variant}")
            # SGD requires its own standard learning rate; this is declared in output rather than hidden.
            if kind == "optimizer" and variant == "sgd_momentum": config.learning_rate=1e-3
            agent=ComparisonDQNAgent(config, optimizer_name=variant if kind=="optimizer" else "adam", activation="relu" if kind=="optimizer" else variant)
            run_dir=output_dir / "training" / f"{kind}_{variant}_seed{seed}"; logger=MetricLogger(run_dir, f"{kind}_{variant}_seed{seed}")
            try: summary=train_agent(YawRLEnvironment(), agent, profile["episodes"], profile["steps_per_episode"], logger, print_interval=0, seed=seed)
            finally: logger.close()
            eval_rows=evaluate_policy(YawRLEnvironment(), lambda state: agent.select_action(state, greedy=True), eval_episodes, seed_base=20_000, measure_inference=True)
            episode_frame=pd.read_csv(run_dir / f"{kind}_{variant}_seed{seed}_episodes.csv")
            step_frame=pd.read_csv(run_dir / f"{kind}_{variant}_seed{seed}_metrics.csv")
            row={"comparison":kind,"variant":variant,"seed":seed,"optimizer":agent.optimizer_name,"activation":agent.activation_name,"learning_rate":config.learning_rate,"train_wall_time_s":summary["wall_time_s"],"convergence_episode":_convergence_episode(episode_frame),"mean_td_loss":float(step_frame.loss.mean()) if not step_frame.empty else float("nan"),"mean_gradient_norm":float(step_frame.grad_norm.mean()) if not step_frame.empty else float("nan")}
            for col in ("episode_reward","average_power_physical","mean_abs_misalignment","yaw_movement_frequency","control_stability","inference_time_ms"):
                row[f"eval_{col}"]=float(np.mean([x[col] for x in eval_rows]))
            rows.append(row)
    raw=pd.DataFrame(rows); raw.to_csv(output_dir / f"{kind}_results.csv", index=False)
    summarise_runs(raw, ["comparison","variant","optimizer","activation","learning_rate"]).to_csv(output_dir / f"{kind}_summary.csv", index=False)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--kind", choices=["optimizer","activation","both"], default="both"); parser.add_argument("--seeds", default="0,1,2,3,4"); parser.add_argument("--profile", choices=PROFILES, default="reduced"); parser.add_argument("--output-dir", default="results/neural_comparisons"); parser.add_argument("--device", default="auto"); parser.add_argument("--eval-episodes", type=int, default=5); args=parser.parse_args()
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    for kind in (["optimizer","activation"] if args.kind=="both" else [args.kind]): run_comparison(kind,seeds,args.profile,Path(args.output_dir),args.device,args.eval_episodes)
if __name__ == "__main__": main()
