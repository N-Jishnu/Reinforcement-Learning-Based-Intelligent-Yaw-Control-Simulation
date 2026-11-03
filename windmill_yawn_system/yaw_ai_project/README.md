# Reinforcement Learning Based Intelligent Yaw Control Simulation

A wind-turbine yaw-control simulation with tabular Q-learning, Deep Q-Network
(DQN), Double DQN, and classical Rule/PID baselines. The upgrade documented
here focuses on Artificial Neural Networks and Deep Learning: architecture
inspection, real training metrics, TensorBoard, ablation experiments,
multi-seed evaluation, tests and documentation.

Simulation -> RL Environment -> Neural-Network Training -> Baselines ->
Dual-Mode Evaluation -> Visualization -> Dashboard -> Documentation

---

## 1) What is in this repository

| Area | Implementation |
|---|---|
| Wind simulation | ERA5-driven wind field with turbulence, gusts and abrupt direction shifts |
| Turbine model | Yaw actuator delay, inertia and rate limits |
| Power model | Normalized mode for RL, physical mode for reporting |
| Baselines | Random, tabular Q-learning, Rule-based, PID |
| Deep RL | Standard DQN and Double DQN in PyTorch |
| Neural network | `5 -> 128 -> 128 -> 3`, ReLU hidden layers, linear output |
| Training metrics | Actual TD loss, actual gradient norm, Q-values, epsilon, timing |
| Experimentation | Ablation study and multi-seed evaluation |
| Dashboard | Streamlit trace view, neural-network analysis, controller comparison |

The simulation, state space, action space and reward function are unchanged.
Existing commands and output files remain valid unless explicitly pointed at
new paths.

---

## 2) System architecture

```text
data/era5_chennai.csv
        |
        v
simulation/wind_field.py   (turbulence + gusts + direction shifts)
        |
        v
simulation/wind_model.py   (seeded wind series)
        |
        v
env/environment.py         (5-D state, 3 actions, yaw dynamics, reward)
        |
        +---------------------------+
        |                           |
        v                           v
tabular Q-learning          PyTorch Q-network
(Q-table)                   (5 -> 128 -> 128 -> 3)
                                    |
                        +-----------+-----------+
                        |                       |
                     DQN                  Double DQN
                (replay + target)   (replay + target + decoupled selection)
                        |                       |
                        +-----------+-----------+
                                    |
                                    v
                evaluation/agent_evaluation.py
        Random | Q-learning | DQN | Double DQN | Rule | PID
                                    |
                                    v
        results/ CSV + results/plots/ PNG + Streamlit dashboard
```

---

## 3) Neural network architecture

| Property | Value |
|---|---|
| Input | 5 normalized state values |
| Hidden layer 1 | 128 neurons, ReLU |
| Hidden layer 2 | 128 neurons, ReLU |
| Output | 3 Q-values (left, right, hold) |
| Total trainable parameters | **17,667** |
| Weight/bias init | PyTorch `Linear` defaults (Kaiming-uniform / uniform) |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Loss | MSE |
| Discount factor | `0.99` |
| Gradient clipping | L2 norm, `5.0` |

Print and visualize the real network:

```bash
python rl/architecture.py --output results/plots/network_architecture.png
```

See [`docs/NEURAL_NETWORK.md`](docs/NEURAL_NETWORK.md) for the full
mathematical and implementation description.

---

## 4) DQN and Double DQN

### Standard DQN target

```text
y = r + gamma * max_a' Q_target(s', a') * (1 - done)
```

### Double DQN target

```text
a* = argmax_a' Q_online(s', a')
y  = r + gamma * Q_target(s', a*) * (1 - done)
```

For terminal transitions the future term is zero. The online network selects
the next action; the target network evaluates it. Experience replay,
epsilon-greedy exploration, target-network synchronisation and gradient
clipping are retained in both algorithms.

The implementation lives in `rl/agents.py`:

- `QNetwork` - the neural network.
- `ReplayBuffer` - fixed-capacity experience replay.
- `DQNAgent` - standard DQN and the no-replay / no-target ablations.
- `DoubleDQNAgent` - Double DQN target rule.

---

## 5) Reward (unchanged)

```python
reward = (
    1.50 * power_normalized
    + 0.80 * alignment_term
    - 0.70 * (misalignment_norm ** 2)
    - 0.06 * abs(yaw_change_applied)
    - 0.30 * switched_direction
    - rapid_flip_penalty
    + alignment_bonus
)
```

with:

- `alignment_term = max(cos(misalignment_rad)^3, 0.0)`
- `rapid_flip_penalty = 0.05` when a direction flip follows a nonzero yaw change
- `alignment_bonus = 0.15` when `abs(misalignment) <= 5 deg`

The reward was not changed during the neural-network upgrade. Training reward
is a shaped control signal; it is not the same quantity as physical energy.

---

## 6) Original hyperparameters (preserved)

- DQN episodes: `2200`
- Q-learning episodes: `500`
- Steps per episode: `1000`
- Replay buffer: `50,000`
- Batch size: `256`
- Train start: after `2,000` transitions
- Target update: every `250` steps
- Epsilon: `1.0 -> 0.05` over `1,500` episodes
- Learning rate: `3e-4`
- Gamma: `0.99`
- Gradient clipping: `5.0`

Reduced-budget experiment profiles override episodes, steps, epsilon schedule,
batch size and buffer size. All algorithms inside a comparison use the same
settings, and reduced runs are labelled in every CSV.

---

## 7) Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `gymnasium`,
`streamlit`, `tensorboard`, `pytest`. CUDA is used automatically when
available; CPU execution is fully supported.

---

## 8) Commands

All commands are run from the project root (`yaw_ai_project/`).

### Train DQN (original command still works)

```bash
python rl/train.py --q-episodes 500 --dqn-episodes 2200 \
  --steps-per-episode 1000 --eval-random-episodes 20
```

This writes `rl/model.pth` and `rl/logs.csv`, as before, and additionally
writes real metrics to `results/training/`.

### Train DQN and Double DQN

```bash
python rl/train.py --algorithm both \
  --q-episodes 500 --dqn-episodes 2200 --steps-per-episode 1000 \
  --metrics-dir results/training \
  --model-path results/models/dqn.pth \
  --double-model-path results/models/double_dqn.pth \
  --tensorboard-dir results/tensorboard
```

### Train Double DQN only

```bash
python rl/train_double_dqn.py \
  --episodes 2200 --steps-per-episode 1000 --seed 42 \
  --tensorboard-dir results/tensorboard
```

### TensorBoard

```bash
tensorboard --logdir results/tensorboard
```

CSV logging continues to work when TensorBoard is disabled. To visualize the
real values from CSV logs produced without TensorBoard:

```bash
python rl/export_tensorboard.py \
  --metrics-dir results/training \
  --tensorboard-dir results/tensorboard \
  --prefixes dqn,double_dqn
```

### Generate training plots

```bash
python rl/visualize_training.py \
  --metrics-dir results/training \
  --plots-dir results/plots
```

### Ablation study

```bash
python rl/experiments.py --mode ablation --profile reduced --seeds 0
```

Variants: `standard_dqn`, `no_replay`, `no_target_network`, `double_dqn`.

### Multi-seed evaluation

```bash
python rl/experiments.py --mode multi_seed --profile reduced \
  --seeds 0,1,2,3,4 --algorithms dqn,double_dqn
```

### Evaluate all six controllers

```bash
python evaluation/agent_evaluation.py \
  --episodes 10 \
  --q-episodes 500 \
  --dqn-model results/models/dqn.pth \
  --double-dqn-model results/models/double_dqn.pth
```

### Legacy comparison (original continuous Rule/PID behavior)

```bash
python evaluation/compare_controllers.py --runs 10 --steps 1000 \
  --model-path rl/model.pth
```

### Legacy DQN vs Random evaluation

```bash
python rl/evaluate.py --episodes 50
```

### Dashboard

```bash
streamlit run dashboard/app.py
```

### Tests

```bash
python -m pytest tests -q
```

---

## 9) Output layout

```text
results/
  training/
    dqn_metrics.csv                 # step-level DQN metrics
    dqn_episodes.csv                # episode-level DQN metrics
    double_dqn_metrics.csv          # step-level Double DQN metrics
    double_dqn_episodes.csv         # episode-level Double DQN metrics
    training_summary.csv            # one row per run
    runs/<run_name>/                # per-experiment logs
  evaluation/
    controller_comparison.csv       # per-episode controller metrics
    controller_comparison_summary.csv
    ablation_results.csv
    ablation_summary.csv
    multi_seed_results.csv
    multi_seed_summary.csv
  models/
    dqn.pth
    double_dqn.pth
    q_table.npz
    <variant>_seed<seed>.pth
  plots/
    network_architecture.png
    dqn_reward.png
    dqn_loss.png
    dqn_epsilon.png
    dqn_q_values.png
    dqn_gradient_norm.png
    dqn_episode_length.png
    training_time_comparison.png
    dqn_vs_double_dqn.png
    ablation_comparison.png
    multi_seed_comparison.png
    controller_comparison.png
  tensorboard/
```

Original artifacts are preserved:

- `rl/model.pth`, `rl/logs.csv`
- `results/logs.csv`, `results/model.pth`
- `results/controller_comparison_runs.csv`,
  `results/controller_comparison_summary.csv`

---

## 10) Results

### 10.1 Original controller results (preserved)

These are the project's original continuous-action comparison results. They
are retained unchanged for provenance. Source:
`results/controller_comparison_summary.csv`.

Normalized average power (mean +/- std):

| Controller | Average power (normalized) | Average power (physical, W) |
|---|---:|---:|
| Rule | 159.652 +/- 49.251 | 393,225 +/- 121,305 |
| PID | 157.750 +/- 50.224 | 388,541 +/- 123,702 |
| RL (original DQN) | 135.216 +/- 36.101 | 333,038 +/- 88,917 |

An earlier evaluation stored in `evaluation/analysis_summary.csv` recorded the
original RL checkpoint at `128.925 +/- 47.137` normalized and
`317,542 +/- 116,100 W`; both legacy files are preserved. The difference comes
from separate legacy evaluation runs and is not hidden.

Control behavior:

| Controller | Yaw movement frequency | Control stability | Mean abs misalignment |
|---|---:|---:|---:|
| Rule | 0.9953 | 0.4650 | 86.236 deg |
| PID | 0.9970 | 0.4758 | 86.777 deg |
| RL | 0.9885 | 0.3862 | 91.450 deg |

Interpretation: the original RL checkpoint reduces yaw movement and control
variance but underperforms Rule/PID in energy capture. This is reported
honestly and is not hidden by the upgrade.

### 10.2 Ablation study (reduced budget, seed 0)

Measured on identical evaluation seeds. Reduced budget: 80 episodes x 300
training steps, not the full 2,200 x 1,000 configuration. Evaluation still uses
complete 1,000-step episodes, so reduced training and evaluation episode lengths
differ; this is one more reason the table is indicative rather than definitive.

| Variant | Eval reward | Norm. power | Physical power (W) | Mean abs misalign. | Yaw freq. | Max grad norm |
|---|---:|---:|---:|---:|---:|---:|
| `standard_dqn` | -17.32 | 119.75 | 294,940 | 94.21 deg | 0.832 | 140.6 |
| `no_replay` | 121.77 | 159.02 | 391,663 | 86.77 deg | 0.349 | 675.6 |
| `no_target_network` | -199.19 | 52.56 | 129,464 | 111.83 deg | 0.719 | 118.7 |
| `double_dqn` | 44.00 | 141.28 | 347,983 | 92.50 deg | 0.842 | 141.0 |

Observations from this single seed:

- Removing replay produced the best evaluation power in this reduced run, but
  the online update is high-variance; one seed is not evidence of superiority.
- Removing the target network was unstable and produced the worst evaluation
  power.
- Double DQN was better than standard DQN in this run, but the difference is
  not established without multiple seeds.
- These reduced-budget results must not be presented as full-training results.

### 10.3 Multi-seed results (reduced budget, 5 seeds)

Measured on identical evaluation seeds (5 eval episodes). Reduced profile:
80 episodes x 300 training steps, 300-step truncation in the training loop while
evaluation runs full 1,000-step episodes. DQN and Double DQN share the same
reduced budget and same seeds.

| Algorithm | Eval reward (mean +/- std) | Norm. power | Physical power (W) | Mean abs misalign. | Yaw freq. | Control stability |
|---|---:|---:|---:|---:|---:|---:|
| `dqn` | -14.675 +/- 95.522 | 116.009 | 285,732 | 95.302 deg | 0.809 | 0.268 |
| `double_dqn` | -104.307 +/- 131.796 | 86.939 | 214,131 | 102.591 deg | 0.889 | 0.314 |

Per-seed eval rewards (seed 0-4): DQN = `-92.362, 77.307, 76.255, -3.266, -131.309`; Double DQN =
`-31.581, 89.975, -185.847, -233.080, -161.003`. Paired difference (Double DQN - DQN):
mean = `-89.63`, std = `146.70`, approximate 95% CI ~ `[-218, +39]` (wide, overlaps zero).
No claim of superiority is made; in this reduced 5-seed run DQN performs better on
average than Double DQN.

Full per-seed table: `results/evaluation/multi_seed_results.csv`. Aggregated
stats: `results/evaluation/multi_seed_summary.csv`.

### 10.4 Six-controller comparison (unified, discrete projection)

Evaluated with `evaluation/agent_evaluation.py` on identical wind traces
(10 episodes). RL agents use checkpoints from the reduced-budget runs above.
Rule/PID are projected onto discrete actions with a `0.5°` dead-band; the
original continuous comparisons are preserved in §10.1.

| Controller | Eval reward (mean +/- std) | Norm. power | Physical power (W) | Mean abs misalign. | Yaw freq. | Time within 10° | Inference (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random` | -63.30 +/- 85.32 | 123.42 | 303,988 | 92.88 deg | 0.996 | 0.049 | 0.000 |
| `q_learning` | -3.22 +/- 95.06 | 131.44 | 323,741 | 91.68 deg | 0.992 | 0.061 | 0.029 |
| `dqn` | -155.95 +/- 119.15 | 71.67 | 176,519 | 106.08 deg | 0.860 | 0.033 | 0.245 |
| `double_dqn` | -111.37 +/- 133.58 | 79.65 | 196,169 | 103.03 deg | 0.828 | 0.039 | 0.241 |
| `rule` | 206.21 +/- 105.24 | 201.91 | 497,314 | 76.19 deg | 0.995 | 0.093 | 0.000 |
| `pid` | 187.60 +/- 108.91 | 198.90 | 489,884 | 76.77 deg | 0.997 | 0.093 | 0.000 |

Note: Q-learning baseline was trained separately (500 episodes x 1,000 steps)
as recorded in `results/models/q_table.npz`; DQN/Double DQN above use reduced
80x300 training. Rule/PID are projected to discrete actions for unified
comparison - different from their original continuous-action baselines. See
`results/evaluation/controller_comparison_summary.csv` for full stats.

---

## 11) Why training reward is not physical power

The reward includes power, alignment shaping, movement penalties, switching
penalties and a small-alignment bonus. A policy can increase reward by holding
still, even when that leaves a large misalignment and reduces physical energy.
Training reward is also measured under epsilon-greedy exploration, while
evaluation is greedy and uses different wind traces. For these reasons,
training and evaluation metrics are reported in separate sections and files.

---

## 12) Experimental limitations

- Full 2,200-episode x 1,000-step multi-seed training is expensive on CPU. The
  multi-seed and ablation tables above use the documented reduced profile.
- Reduced-budget rankings are indicative, not definitive.
- Rule and PID are projected onto the discrete action set for the unified
  comparison; the original continuous results are preserved separately.
- CUDA determinism is best-effort. Exact bitwise reproduction across different
  GPUs is not guaranteed.
- No claim is made that Double DQN outperforms DQN unless the measured
  multi-seed results support it.
- The PPO/farm/wake code in the repository is a separate track and is not part
  of the main DQN comparison.

---

## 13) Repository structure

```text
simulation/     wind field, wind series, turbine and power models
env/            Gym-style yaw environment
rl/             agents, trainer, metrics, training, experiments, visualization
evaluation/     controller comparison and legacy evaluation scripts
dashboard/      Streamlit dashboard
tests/          pytest suite
docs/           technical documentation
results/        generated metrics, summaries, models and plots
data/           ERA5 wind dataset
visualization/  legacy plots
paper/          paper draft
```

---

## 14) Tests

```bash
python -m pytest tests -q
```

The suite covers:

- Network input/output shapes and parameter counts.
- Replay buffer add, sample and capacity.
- DQN action selection and exploration.
- DQN training update and real metric values.
- Standard and Double DQN target calculations.
- Terminal-transition handling.
- Target-network synchronisation.
- Checkpoint save/load and hidden-size inference.
- Seeded reproducibility of networks, actions and environment resets.
- CSV metric logging.

The exact test count and result are reported in the final change summary.

---

## 15) Documentation

- [`docs/NEURAL_NETWORK.md`](docs/NEURAL_NETWORK.md) - full neural-network,
  DQN, Double DQN, metrics, ablation and reproducibility documentation.
- `paper/document` - original paper draft.
