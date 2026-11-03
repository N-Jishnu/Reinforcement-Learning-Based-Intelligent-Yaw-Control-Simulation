"""Train/evaluate LSTM-DQN and Attention-LSTM-DQN in separate result folders."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from env.environment import YawRLEnvironment
from rl.agents import DQNAgent, set_global_seed
from rl.experiments import PROFILES, summarise_runs
from rl.temporal_agents import LSTMDQNAgent, TemporalDQNConfig
from rl.temporal_trainer import evaluate_temporal_agent, train_temporal_agent

def run(algorithms, seeds, profile_name, output_dir, device, eval_episodes):
    profile=dict(PROFILES[profile_name]); output_dir=Path(output_dir); rows=[]
    for name in algorithms:
        for seed in seeds:
            set_global_seed(seed); attention=name == "attention_lstm_dqn"
            config=TemporalDQNConfig(sequence_length=20, batch_size=profile["batch_size"], buffer_size=profile["buffer_size"], train_start_steps=profile["train_start_steps"], target_update_every=profile["target_update_every"], epsilon_decay_episodes=profile["epsilon_decay_steps"], device=device, attention=attention, name=name)
            agent=LSTMDQNAgent(config); run_dir=output_dir / "training" / f"{name}_seed{seed}"
            summary=train_temporal_agent(YawRLEnvironment(),agent,profile["episodes"],profile["steps_per_episode"],run_dir,name,seed)
            checkpoint=output_dir / "models" / f"{name}_seed{seed}.pth"; agent.save(checkpoint)
            attention_path=output_dir / "attention" / f"{name}_seed{seed}.csv" if attention else None
            if attention_path: attention_path.parent.mkdir(parents=True,exist_ok=True)
            evaluation=evaluate_temporal_agent(YawRLEnvironment(),agent,eval_episodes,20_000,attention_path)
            row={"algorithm":name,"seed":seed,"sequence_length":20,"train_wall_time_s":summary["wall_time_s"],"train_mean_td_loss":summary["mean_loss"],"train_mean_gradient_norm":summary["mean_grad_norm"],"checkpoint":str(checkpoint)}
            for metric in ("episode_reward","average_power_normalized","average_power_physical","mean_abs_misalignment","yaw_movement_frequency","control_stability","inference_time_ms"):
                row[f"eval_{metric}"]=float(np.mean([x[metric] for x in evaluation])); row[f"eval_{metric}_std_episode"]=float(np.std([x[metric] for x in evaluation],ddof=1)) if len(evaluation)>1 else 0.0
            rows.append(row)
    raw=pd.DataFrame(rows); evaluation_dir=output_dir / "evaluation"; evaluation_dir.mkdir(parents=True,exist_ok=True)
    raw.to_csv(evaluation_dir / "temporal_results.csv",index=False); summarise_runs(raw,["algorithm","sequence_length"]).to_csv(evaluation_dir / "temporal_summary.csv",index=False)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--algorithms",default="lstm_dqn,attention_lstm_dqn"); parser.add_argument("--seeds",default="0,1,2,3,4"); parser.add_argument("--profile",choices=PROFILES,default="reduced"); parser.add_argument("--output-dir",default="results/temporal_experiments"); parser.add_argument("--device",default="auto"); parser.add_argument("--eval-episodes",type=int,default=5); args=parser.parse_args()
    run([x.strip() for x in args.algorithms.split(',') if x.strip()],[int(x) for x in args.seeds.split(',') if x.strip()],args.profile,args.output_dir,args.device,args.eval_episodes)
if __name__ == '__main__': main()
