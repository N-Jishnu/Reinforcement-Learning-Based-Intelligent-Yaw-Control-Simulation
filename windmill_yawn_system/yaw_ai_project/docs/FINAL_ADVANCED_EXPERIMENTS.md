# Final advanced neural-network experiments

## Scope and provenance

This report documents the complete imported Colab bundle at
`results/colab_final_advanced_experiment/`. It is separate from all legacy and
local partial outputs. Each reported algorithm was trained with five random
seeds using the reduced budget (80 episodes x 300 interaction steps), then
evaluated greedily on five seeded wind episodes. The statistics below are means
and sample standard deviations across seeds, not fabricated or selected runs.

## Temporal architectures

| Algorithm | Mean evaluation reward | Reward SD | Mean physical power | Power SD | Mean TD loss | Mean train time |
|---|---:|---:|---:|---:|---:|---:|
| LSTM-DQN | 14.71 | 68.86 | 314.3 kW | 49.2 kW | 1.149 | 99.69 s |
| Attention-LSTM-DQN | -7.86 | 46.70 | 286.8 kW | 51.8 kW | 1.292 | 113.39 s |

For this reduced-budget experiment, ordinary LSTM-DQN had higher mean reward
and physical power than Attention-LSTM-DQN. This result does **not** establish
that attention is generally harmful or that LSTM is asymptotically better;
confidence intervals are broad and the budget is intentionally limited.

Attention evaluation data is in
`temporal_experiments/attention/`; associated heatmaps are in `plots/`.

## Optimizer comparison

All runs used the same feedforward architecture and ReLU. Adam and RMSProp use
learning rate `3e-4`; SGD with momentum uses the explicitly recorded `1e-3`.

| Optimizer | Mean evaluation reward | Mean physical power | Mean TD loss | Mean gradient norm | Mean train time |
|---|---:|---:|---:|---:|---:|
| Adam | -57.95 | 255.6 kW | 5.492 | 28.09 | 72.50 s |
| RMSProp | -52.77 | 258.7 kW | 5.476 | 28.16 | 70.84 s |
| SGD + momentum | 3.27 | 303.5 kW | 4.563 | 15.77 | 67.82 s |

SGD with momentum achieved the highest mean reward and physical power in this
specific reduced-budget comparison. It should not be treated as a universal
optimizer ranking.

## Activation comparison

All runs used Adam at `3e-4`, the same `5 -> 128 -> 128 -> 3` topology, and a
linear Q-value output layer.

| Hidden activation | Mean evaluation reward | Mean physical power | Mean TD loss | Mean gradient norm |
|---|---:|---:|---:|---:|
| ReLU | -57.95 | 255.6 kW | 5.492 | 28.09 |
| Leaky ReLU | -15.18 | 289.2 kW | 5.589 | 29.09 |
| Tanh | -41.53 | 275.2 kW | 5.271 | 13.47 |

Leaky ReLU produced the strongest mean reward and power in this run, while
Tanh had the lowest mean TD loss and gradient norm. These are different
criteria; no activation is declared globally best.

## Reproducing on Colab

Select a GPU runtime, clone a revision that includes the temporal and
comparison modules, install non-Torch dependencies, and run:

```bash
python rl/temporal_experiments.py --algorithms lstm_dqn,attention_lstm_dqn \
  --seeds 0,1,2,3,4 --profile reduced --device cuda --eval-episodes 5
python rl/nn_comparisons.py --kind both --seeds 0,1,2,3,4 \
  --profile reduced --device cuda --eval-episodes 5
```

Then invoke `rl/visualize_nn_comparisons.py` to create comparative plots. CPU
runs are substantially slower; the local LSTM runs took roughly 18--21 minutes
per seed in the observed environment.

## Limitations

- Results use the reduced profile, not the 2,200-episode full profile.
- Five seeds provide useful variability estimates but wide intervals remain.
- Physical power, training reward and TD loss are distinct objectives and must
  not be substituted for one another.
- LSTM/attention models are evaluated on the same seeded protocol within their
  own study; cross-study conclusions should account for training budget and
  checkpoint provenance.
