"""Merge independently generated controller, feedforward and temporal results.

This never rewrites source experiment CSVs; it only creates a new unified
table when the corresponding source artifacts exist.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def _read(path: Path):
    try: return pd.read_csv(path) if path.exists() else None
    except Exception: return None

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base-dir',default='results'); parser.add_argument('--output-dir',default='results/unified_evaluation'); args=parser.parse_args()
    base=Path(args.base_dir); rows=[]
    # Prior controller experiment has its own naming convention; normalize it.
    controllers=_read(base/'final_experiment'/'evaluation'/'controller_comparison_summary.csv')
    if controllers is None:
        controllers=_read(base/'evaluation'/'controller_comparison_summary.csv')
    if controllers is not None:
        for _,r in controllers.iterrows(): rows.append({'algorithm':str(r['controller']),'source':'controller_evaluation','mean_cumulative_reward':r.get('episode_reward_mean'),'reward_std':r.get('episode_reward_std'),'mean_physical_power':r.get('average_power_physical_mean'),'power_std':r.get('average_power_physical_std'),'mean_yaw_misalignment':r.get('mean_abs_misalignment_mean'),'yaw_movement_frequency':r.get('yaw_movement_frequency_mean'),'control_stability':r.get('control_stability_mean'),'inference_time_ms':r.get('inference_time_ms_mean')})
    for source,path in [('feedforward',base/'final_experiment'/'evaluation'/'multi_seed_summary.csv'),('temporal',base/'temporal_experiments'/'evaluation'/'temporal_summary.csv')]:
        frame=_read(path)
        if frame is None: continue
        for _,r in frame.iterrows(): rows.append({'algorithm':r.get('algorithm'),'source':source,'mean_cumulative_reward':r.get('eval_episode_reward_mean'),'reward_std':r.get('eval_episode_reward_std'),'mean_physical_power':r.get('eval_average_power_physical_mean'),'power_std':r.get('eval_average_power_physical_std'),'mean_td_loss':r.get('train_mean_td_loss_mean'),'mean_yaw_misalignment':r.get('eval_mean_abs_misalignment_mean'),'yaw_movement_frequency':r.get('eval_yaw_movement_frequency_mean'),'control_stability':r.get('eval_control_stability_mean'),'training_time_s':r.get('train_wall_time_s_mean'),'inference_time_ms':r.get('eval_inference_time_ms_mean')})
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output/'unified_comparison.csv',index=False)
    print(f'Saved {output / "unified_comparison.csv"} with {len(rows)} rows')
if __name__ == '__main__': main()
