"""Plot optimizer, activation, temporal and attention CSV outputs without synthetic data."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def plot_bars(frame, label, columns, output):
    if frame is None or frame.empty: return
    fig, axes=plt.subplots(1,len(columns),figsize=(5*len(columns),4)); axes=[axes] if len(columns)==1 else axes
    for ax,(column,title) in zip(axes,columns):
        mean=f"{column}_mean" if f"{column}_mean" in frame else column; std=f"{column}_std" if f"{column}_std" in frame else None
        if mean not in frame: continue
        ax.bar(frame[label],frame[mean],yerr=frame[std] if std in frame else None,capsize=3); ax.set_title(title); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.3)
    fig.tight_layout(); output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=180); plt.close(fig)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--comparison-dir',default='results/neural_comparisons'); parser.add_argument('--temporal-dir',default='results/temporal_experiments'); parser.add_argument('--output-dir',default='results/neural_comparisons/plots'); args=parser.parse_args(); output=Path(args.output_dir)
    for kind in ('optimizer','activation'):
        path=Path(args.comparison_dir)/f'{kind}_summary.csv'
        if path.exists(): plot_bars(pd.read_csv(path),'variant',[('eval_episode_reward','Evaluation reward'),('eval_average_power_physical','Physical power (W)'),('mean_td_loss','TD loss'),('convergence_episode','Convergence episode')],output/f'{kind}_comparison.png')
    temporal=Path(args.temporal_dir)/'evaluation'/'temporal_summary.csv'
    if temporal.exists(): plot_bars(pd.read_csv(temporal),'algorithm',[('eval_episode_reward','Evaluation reward'),('eval_average_power_physical','Physical power (W)'),('train_mean_td_loss','TD loss'),('train_wall_time_s','Training time (s)')],output/'temporal_comparison.png')
    attention_dir=Path(args.temporal_dir)/'attention'
    for path in attention_dir.glob('*.csv') if attention_dir.exists() else []:
        frame=pd.read_csv(path); lag=[c for c in frame if c.startswith('lag_')]
        if lag:
            fig,ax=plt.subplots(figsize=(10,4)); ax.imshow(frame[lag].head(100).T,aspect='auto',cmap='viridis'); ax.set_xlabel('Decision step'); ax.set_ylabel('History position'); ax.set_title('Temporal attention weights'); fig.tight_layout(); fig.savefig(output/f'{path.stem}_attention.png',dpi=180); plt.close(fig)
if __name__ == '__main__': main()
