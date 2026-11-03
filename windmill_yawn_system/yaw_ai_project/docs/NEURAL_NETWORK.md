# Neural Network Implementation

This document explains the Artificial Neural Network (ANN) and Deep
Reinforcement Learning implementation used by the wind-turbine yaw controller.
It describes the code that exists in this repository, not a proposed design.

---

## 1. Role of the neural network

The yaw controller is a discrete-action Reinforcement Learning problem. The
neural network is a **Q-network**: it estimates the expected discounted return
of each of the three possible yaw actions for the current state.

```text
state s (5 values)  ->  Q-network  ->  Q(s, left), Q(s, hold), Q(s, right)
```

The controller then selects the action with the highest estimated value during
evaluation, or an epsilon-greedy action during training.

The network is deliberately small and fully connected. The state is a compact
set of five physical measurements, so a convolutional or recurrent network
would add complexity without a physical justification.

---

## 2. Architecture

The architecture is unchanged from the original project.

| Property | Value |
|---|---|
| Input size | 5 |
| Hidden layer 1 | 128 neurons, ReLU |
| Hidden layer 2 | 128 neurons, ReLU |
| Output size | 3 (linear) |
| Total trainable parameters | 17,667 |
| Weight initialisation | PyTorch `Linear` default (Kaiming-uniform) |
| Bias initialisation | PyTorch `Linear` default (uniform) |
| Optimizer | Adam |
| Learning rate | `3e-4` |
| Loss | Mean squared error (MSE) |
| Discount factor | `0.99` |
| Gradient clipping | L2 norm, `5.0` |

Parameter count:

```text
Linear(5, 128)     = 5 * 128 + 128       =    768
Linear(128, 128)   = 128 * 128 + 128     = 16,512
Linear(128, 3)     = 128 * 3 + 3         =    387
Total                                    = 17,667
```

The code that defines the network is in `rl/agents.py`
(`QNetwork`). `QNetwork` is also re-exported from `rl/train.py` for backward
compatibility with older scripts.

Print and visualize the real architecture:

```bash
python rl/architecture.py --output results/plots/network_architecture.png
```

The command prints layer shapes, activations and parameter counts, then draws
the actual topology with Matplotlib. The function
`QNetwork.parameter_summary()` is the single source of truth used by the
dashboard and reports.

---

## 3. Forward propagation

For a mini-batch of states `S` with shape `(batch, 5)`:

```text
h1 = ReLU(S  W1 + b1)
h2 = ReLU(h1 W2 + b2)
Q  = h2 W3 + b3
```

`Q` has shape `(batch, 3)`. Each row contains the estimated value of the three
discrete actions. The implementation uses `torch.nn.Sequential` so that the
mathematical operations are directly visible in `QNetwork.net`.

---

## 4. Loss and backpropagation

For a sampled transition `(s, a, r, s', done)`, the online network produces
`Q_online(s, a)`. The bootstrap target depends on the algorithm.

### Standard DQN

```text
y = r + gamma * max_a' Q_target(s', a') * (1 - done)
```

### Double DQN

```text
a* = argmax_a' Q_online(s', a')
y  = r + gamma * Q_target(s', a*) * (1 - done)
```

For terminal transitions `done = 1`, the future term is exactly zero.

The loss is computed only on the selected action:

```text
L = (1 / N) * sum_i ( Q_online(s_i, a_i) - y_i )^2
```

Backpropagation and the optimizer step are explicit in
`DQNAgent._optimize_batch()`:

```python
loss = nn.functional.mse_loss(current_q, target_q)
self.optimizer.zero_grad()
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), clip)
self.optimizer.step()
```

The reported loss is the actual tensor returned by `mse_loss`. The reported
gradient norm is the value returned by `clip_grad_norm_` (the norm *before*
clipping). No synthetic or dummy values are logged.

---

## 5. Experience replay

`ReplayBuffer` stores transitions in a fixed-capacity `collections.deque`:

```text
(s, a, r, s', done)
```

Default capacity for the original training run: `50,000` transitions.
A mini-batch of `256` is sampled uniformly at random. Sampling returns NumPy
arrays, which are moved to the compute device in one operation.

Why replay matters:

- Breaks the strong temporal correlation between consecutive simulator steps.
- Reuses each transition multiple times.
- Stabilises the gradient signal.

The ablation variant `no_replay` disables the buffer and performs an online
update with a batch of one transition. This is intentionally high-variance and
is expected to be less stable; the experiment records the outcome rather than
forcing success.

---

## 6. Target network

A separate copy of the online network is used to compute bootstrapped targets:

```python
self.target = copy.deepcopy(self.online)
self.target.eval()
```

The target network is synchronised by a hard copy every `250` environment
steps in the original configuration:

```python
self.target.load_state_dict(self.online.state_dict())
```

The ablation variant `no_target_network` computes targets directly from the
online network. This removes the moving-target stabilisation mechanism and can
produce unstable or divergent Q-values; the experiment reports this honestly.

---

## 7. Exploration and exploitation

Training uses an epsilon-greedy policy:

```text
with probability epsilon:  choose a random action
otherwise:                 choose argmax_a Q_online(s, a)
```

The original schedule is linear from `1.0` to `0.05` over the first `1,500`
episodes, then constant at `0.05`.

Evaluation always uses `greedy=True`; epsilon is not used for reported
controller metrics.

---

## 8. Environment and reward (unchanged)

The neural network operates on the existing `YawRLEnvironment`.

### State (5-D, normalized to `[0, 1]`)

1. Wind direction `/ 360`
2. Wind speed normalized from `[3, 15] m/s`
3. Yaw angle `/ 360`
4. Misalignment normalized from `[-180, 180]`
5. Wind-direction change normalized from `[-180, 180]`

### Actions (discrete, 3)

- `0` = yaw left (`-2 deg` command)
- `1` = yaw right (`+2 deg` command)
- `2` = hold

### Reward (unchanged)

```text
reward = 1.50 * power_normalized
       + 0.80 * alignment_term
       - 0.70 * (misalignment_norm ** 2)
       - 0.06 * abs(yaw_change_applied)
       - 0.30 * switched_direction
       - rapid_flip_penalty
       + alignment_bonus
```

The reward was not modified during this upgrade. Training reward is a shaped
control signal, not a direct measure of physical energy production.

---

## 9. Training metrics

Metrics are separated by frequency so that the CSV files remain manageable.

### Step-level metrics (`*_metrics.csv`)

Written every `log_interval` optimisation steps:

| Column | Meaning |
|---|---|
| `episode` | Episode in which the update occurred |
| `global_step` | Environment step counter |
| `loss` | Actual MSE TD loss from the update |
| `grad_norm` | Actual L2 gradient norm before clipping |
| `avg_q` | Mean Q-value of the selected actions in the batch |
| `max_q` | Maximum selected Q-value in the batch |
| `epsilon` | Current exploration rate |
| `learning_rate` | Optimizer learning rate |

### Episode-level metrics (`*_episodes.csv`)

Written once per episode:

`episode_reward`, `moving_avg_50`, `epsilon`, `episode_length`,
`episode_time_s`, `total_time_s`, `optimization_steps`, `mean_loss`,
`mean_grad_norm`, `mean_q`, `max_q`, `learning_rate`.

### Run-level summary (`training_summary.csv`)

One appended row per run with algorithm, seed, total steps, optimisation steps,
wall-clock time, final/best reward, mean loss, mean gradient norm, parameter
count, device and timestamp.

CSV logging is independent of TensorBoard. TensorBoard is optional and is only
created when `--tensorboard-dir` is provided.

If a run was executed without TensorBoard, the real CSV values can be exported
later:

```bash
python rl/export_tensorboard.py \
  --metrics-dir results/training \
  --tensorboard-dir results/tensorboard \
  --prefixes dqn,double_dqn
```

This mirrors the CSV values into event files; it does not invent or interpolate
data.

---

## 10. TensorBoard

Enable logging:

```bash
python rl/train.py \
  --algorithm both \
  --dqn-episodes 2200 \
  --steps-per-episode 1000 \
  --tensorboard-dir results/tensorboard
```

Launch:

```bash
tensorboard --logdir results/tensorboard
```

The tags are grouped as `train/*` (loss, gradient norm, Q-values, epsilon,
learning rate) and `episode/*` (reward, moving average, epsilon, length, mean
loss/Q-values).

---

## 11. Double DQN implementation

`DoubleDQNAgent` subclasses `DQNAgent` and sets `double_dqn=True`. The target
calculation is handled in `DQNAgent.compute_target()`:

```python
next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
next_q = self.target(next_states).gather(1, next_actions)
target = rewards + gamma * next_q * (1.0 - dones)
```

Key properties:

- The **online** network chooses the next action.
- The **target** network evaluates that action.
- Terminal transitions do not bootstrap.
- Replay, epsilon-greedy exploration and target synchronisation are retained.

Standard DQN and Double DQN share the same network, replay buffer, optimiser,
discount, learning rate, batch size, gradient clipping and evaluation
environment. Only the target rule differs.

Train Double DQN directly:

```bash
python rl/train_double_dqn.py \
  --episodes 2200 \
  --steps-per-episode 1000 \
  --seed 42 \
  --tensorboard-dir results/tensorboard
```

---

## 12. Ablation study

`rl/experiments.py --mode ablation` trains four variants with identical
environment, network, optimiser and evaluation seeds:

| Variant | Replay | Target network | Double target |
|---|---|---|---|
| `standard_dqn` | yes | yes | no |
| `no_replay` | no | yes | no |
| `no_target_network` | yes | no | no |
| `double_dqn` | yes | yes | yes |

Recorded per run:

- Final and best training reward.
- Mean TD loss and loss standard deviation.
- Maximum gradient norm.
- A conservative divergence flag and loss-growth ratio.
- Evaluation reward, normalized and physical power.
- Mean absolute misalignment, yaw movement frequency, control stability,
  time within 10 degrees and inference time.
- Wall-clock training time.

Run:

```bash
python rl/experiments.py --mode ablation --profile reduced --seeds 0
```

The `full` profile reproduces the original 2,200-episode x 1,000-step budget.
The `reduced` profile is used when CPU time is limited and is labelled
`reduced budget` in every output row.

Important detail: the environment itself always defines a 1,000-step episode.
In the reduced profile the *training loop* stops after 300 steps, while
evaluation still runs complete 1,000-step episodes. This is intentional (it
keeps evaluation comparable to the original experiment) but means training and
evaluation episode lengths differ in reduced runs; it is part of why reduced
results are indicative only. A full-profile run uses 2,200 x 1,000 training
steps and matches the original episode length.

---

## 13. Multi-seed experiments

```bash
python rl/experiments.py --mode multi_seed --profile reduced \
  --seeds 0,1,2,3,4 --algorithms dqn,double_dqn
```

Reproducibility controls:

- Python `random`
- NumPy global RNG
- PyTorch CPU and CUDA RNG
- Per-episode environment seed (`seed + episode`)
- Fixed evaluation seeds shared by every controller

CUDA note: `torch.backends.cudnn.deterministic = True` and
`benchmark = False` are set. Some CUDA kernels can still be non-deterministic,
and floating-point reduction order can differ across hardware. Exact bitwise
reproduction across different GPUs is therefore not guaranteed.

Aggregation reports mean, standard deviation and an approximate 95% confidence
interval (`1.96 * s / sqrt(n)`) for each metric.

---

## 14. Evaluation protocol

`evaluation/agent_evaluation.py` evaluates all controllers on the **same**
seeded wind traces. Every controller interacts with the same
`YawRLEnvironment`, so actuator delay, inertia and yaw-rate limits are
identical.

Controllers:

1. Random
2. Tabular Q-learning
3. Standard DQN
4. Double DQN
5. Rule-based
6. PID

Rule and PID normally produce continuous yaw commands. For the discrete-action
comparison their command is projected onto `left / right / hold` with a
`0.5 deg` dead-band. The original continuous Rule/PID results remain available
in `results/controller_comparison_summary.csv` and are preserved.

Evaluation metrics:

- Cumulative and average reward.
- Average normalized power.
- Average physical power (W).
- Total normalized and physical energy.
- Mean absolute yaw misalignment.
- Yaw movement frequency.
- Control stability (standard deviation of applied yaw change).
- Time within 10 degrees of the wind.
- Cumulative yaw movement and reversals.
- Model inference time (measured for Q-learning/DQN/Double DQN; reported as
  `0` for Random, Rule and PID because inference timing is not applicable).

---

## 15. Why training reward is not physical power

Training reward combines normalized power with alignment shaping, movement
penalties, switching penalties and a small-alignment bonus. Two policies can
have similar training reward while producing different physical energy because:

1. Reward is shaped; physical power is not.
2. The reward is discounted within an episode, while reported energy is summed.
3. A policy can reduce movement penalties by holding, even when that leaves a
   large misalignment and lowers power.
4. Training episodes and evaluation episodes use different wind traces.
5. Training reward is measured under epsilon-greedy exploration, while
   evaluation is greedy.

For these reasons, the project reports training and evaluation metrics
separately and does not use training reward as evidence of controller quality.

---

## 16. File map

| File | Purpose |
|---|---|
| `rl/agents.py` | Q-network, replay buffer, DQN and Double DQN agents, seeding |
| `rl/trainer.py` | Shared training loop and evaluation metrics |
| `rl/metrics.py` | CSV and optional TensorBoard logging |
| `rl/train.py` | Original training entry point, extended with DQN/Double DQN |
| `rl/train_double_dqn.py` | Dedicated Double DQN entry point |
| `rl/experiments.py` | Ablation and multi-seed experiments |
| `rl/architecture.py` | Architecture printout and diagram |
| `rl/export_tensorboard.py` | Mirror real CSV metrics into TensorBoard events |
| `rl/visualize_training.py` | Training, ablation, multi-seed and controller plots |
| `evaluation/agent_evaluation.py` | Six-controller evaluation on shared wind traces |
| `tests/test_rl_core.py` | Network, replay, target, checkpoint and logging tests |
| `dashboard/app.py` | Streamlit training analysis and controller comparison |

---

## 17. Limitations

- The reduced-budget experiments are not a substitute for full 2,200-episode
  training. They are labelled as reduced and should not be presented as full
  results.
- Single-seed ablations are indicative only. Multi-seed ablations require
  additional compute.
- Rule and PID are projected onto discrete actions in the unified comparison,
  which changes their behaviour relative to the legacy continuous evaluation.
- The Q-network is small; this is appropriate for a 5-D state but limits
  representational capacity.
- CUDA determinism is best-effort, not guaranteed across hardware.
- No claim is made that Double DQN outperforms DQN unless the measured
  multi-seed results support it.

---

## 18. Temporal and controlled-comparison extensions

The original feedforward network remains `5 -> 128 -> 128 -> 3`, with a linear
Q-value output. Additional experiments are implemented in separate modules so
their configuration cannot alter the original DQN or Double-DQN results.

### LSTM-DQN

`rl/temporal_agents.py` accepts a causal history of 20 existing five-feature
observations. It processes `[batch, time, features]` tensors with `nn.LSTM`,
passes the last hidden representation to a linear three-action Q head, and
uses independent online and target networks. Histories are reset per episode,
left-padded only with the initial observation, and saved atomically in replay;
therefore they cannot include observations from a previous episode or future
wind states. TD targets include `(1 - done)`, so terminal sequences never
bootstrap. Gradient clipping, checkpoint save/load, TD loss and gradient-norm
logging are retained.

### Attention-LSTM-DQN

The optional attention model scores every LSTM output over the sequence,
normalizes scores with softmax across time, and feeds the weighted temporal
representation to the same linear Q head. Evaluation writes attention weights
to CSV and the plot utility renders heatmaps. Attention weights indicate where
the network placed relative emphasis; they are not evidence of an improvement
unless evaluation results demonstrate one.

### Optimizer and activation controls

`rl/nn_comparisons.py` keeps the feedforward topology, replay, target network,
environment, reward, batch size and evaluation protocol fixed. Optimizer runs
compare Adam and RMSProp at `3e-4` with SGD+momentum at its explicitly recorded
`1e-3` rate. Activation runs compare ReLU, Leaky ReLU and Tanh while keeping a
linear output head. CSVs record TD loss, gradient norm, reward, physical power,
misalignment, control stability, inference time and wall-clock time.

See `LSTM_DQN.md` for gate-level explanations and
`FINAL_ADVANCED_EXPERIMENTS.md` for imported five-seed results.
