# Reinforcement Learning Based Intelligent Yaw Control Simulation

This repository implements a full wind-turbine yaw-control pipeline:

Simulation -> RL Environment -> RL Training -> Baselines -> Dual-Mode Evaluation -> Visualization -> Dashboard -> Paper Draft

---

## 1) Project Status

### Completed

- Wind and turbine simulation
  - Wind generator with directional noise and abrupt shifts
  - Turbine yaw dynamics with delay, inertia, and yaw-rate constraints
  - Dual-mode power model (normalized + physical)
- RL environment
  - Gym-style environment with normalized 5D state and discrete 3-action policy
  - 1000-step episodes, normalized-power reward
- RL training
  - Q-learning baseline + DQN implementation
  - Replay buffer, target network updates, logging, model save
- Baseline comparison and evaluation
  - Rule, PID, RL compared over 10 runs x 1000 steps
  - Metrics in normalized and physical modes
  - Statistical summary generation
- Visualization and documentation
  - Training/evaluation plots
  - Streamlit dashboard
  - Paper draft document with required sections

---

## 2) Repository Structure

- `simulation/`
  - `wind_field.py` - wind disturbances and real-data driven generator
  - `wind_model.py` - array outputs + simulation loop helper
  - `turbine_model.py` - yaw actuator dynamics
  - `power_model.py` - `compute_power(..., mode)` for normalized/physical
- `env/`
  - `environment.py` - RL environment (`YawRLEnvironment`)
- `rl/`
  - `train.py` - Q-learning + DQN training pipeline
  - `evaluate.py` - DQN vs random policy evaluation
  - `plot_training.py` - training convergence plots
  - `model.pth` - trained model output
  - `logs.csv` - training logs output
- `evaluation/`
  - `compare_controllers.py` - Rule/PID/RL comparative runs (10x1000 default)
  - `plot_controller_comparison.py` - summary plots + mechanics CSV
- `visualization/`
  - `generate_plots.py`
  - `plots/` - generated figures
- `dashboard/`
  - `app.py` - Streamlit demo
- `paper/`
  - `document` - research-paper draft content

---

## 3) Final Execution Flow

1. Simulation -> RL Environment -> RL Training
2. Simulation -> Baselines
3. RL Model + Baselines -> Evaluation (normalized + physical)
4. Evaluation -> Visualization + Dashboard + Paper

---

## 4) Core Technical Configuration

### Wind and Physics

- Timestep: `1.0 s`
- Wind speed range: `[3, 15] m/s`
- Direction noise: Gaussian, std `3 deg`
- Direction shifts: every `50-100` steps
- Yaw max rate: `2 deg/s`
- Yaw inertia: `0.15`
- Actuation delay: `3 steps`

### RL Environment

- State (normalized):
  - `[wind_direction, wind_speed, yaw_angle, misalignment, wind_direction_change]`
- Action space (discrete):
  - `0=left`, `1=right`, `2=hold` (command step `2 deg`)
- Episode length: `1000 steps`

### Reward (latest version)

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
- `rapid_flip_penalty = 0.05` if direction flips while previous yaw change was nonzero
- `alignment_bonus = 0.15` if `abs(misalignment) <= 5 deg`

### Training (latest target config)

- DQN episodes: `2200`
- Q-learning episodes: `500`
- Steps per episode: `1000`
- DQN learning rate: `3e-4`
- Gamma: `0.99`
- Epsilon: `1.0 -> 0.05`, decay over first `1500` episodes
- Replay buffer: `50000`
- Batch size: `256`
- Train start after: `2000` transitions
- Target update every: `250` steps
- Gradient clipping: `5.0`

---

## 5) Dual-Mode Power Model

### Normalized mode (for RL)

\[
P = 0.5 \cdot v^3 \cdot \cos(\theta)^3
\]

### Physical mode (for reporting)

\[
P = 0.5 \cdot \rho \cdot A \cdot C_p \cdot v^3 \cdot \cos(\theta)^3
\]

Constants:

- `rho = 1.225 kg/m^3`
- `Cp = 0.4`
- `R = 40 m`
- `A = pi * R^2`

---

## 6) Latest Recorded Results

### Kaggle Full Training Run (reported)

- Device: `cuda (Tesla T4)`
- Random baseline reward: `37.695`
- Q-learning Avg(last 50): `55.056`
- DQN Avg(last 50): `38.590`
- DQN improvement over random (train summary): `+0.895`

### Kaggle Evaluation (`rl/evaluate.py --episodes 50`)

- Random mean: `45.766 +/- 24.243`
- DQN mean: `51.636 +/- 33.067`
- Improvement: `+5.870`
- Result: `PASS (DQN better than random)`

### Baseline Comparison (current local `evaluation/compare_controllers.py`)

Normalized average power (mean +/- std):

- Rule: `159.652 +/- 51.915`
- PID: `157.750 +/- 52.941`
- RL: `135.216 +/- 36.101`

Physical average power (mean +/- std, W):

- Rule: `393225.288 +/- 127867.110`
- PID: `388540.507 +/- 130393.476`
- RL: `333038.150 +/- 88916.662`

Control behavior (mean):

- Yaw movement frequency: Rule `0.9953`, PID `0.9970`, RL `0.9885`
- Control stability (std yaw change): Rule `0.4650`, PID `0.4758`, RL `0.3862`
- Mean absolute misalignment (deg): Rule `86.236`, PID `86.777`, RL `91.450`

Note: RL currently reduces movement but underperforms PID/Rule in energy on local baseline outputs. Reward/training updates were applied to improve this in the next retrain cycle.

### Latest retrain status (new config run)

From the current `rl/logs.csv` generated with updated reward/training configuration:

- Logged rows: `2700`
  - Q-learning episodes: `500`
  - DQN episodes: `2200`
- Epsilon schedule verified in logs:
  - starts at `1.0`
  - decays linearly to about `0.05` by episode `1500`
  - remains `0.05` thereafter
- Current DQN rolling reward at final episode:
  - `average_reward_50 = 150.567`

Interpretation: training converges in reward terms, but controller-level evaluation still shows RL below PID on energy.

---

## 7) How to Run

### Train

```bash
python rl/train.py --q-episodes 500 --dqn-episodes 2200 --steps-per-episode 1000 --eval-random-episodes 20
```

### Evaluate RL vs Random

```bash
python rl/evaluate.py --episodes 50
```

### Run Baselines (Rule/PID/RL)

```bash
python evaluation/compare_controllers.py --runs 10 --steps 1000 --model-path rl/model.pth
```

### Generate Plots

```bash
python visualization/generate_plots.py
python evaluation/plot_controller_comparison.py
python rl/plot_training.py
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 8) Output Artifacts

- Model and logs:
  - `rl/model.pth`
  - `rl/logs.csv`
- Evaluation:
  - `results/controller_comparison_runs.csv`
  - `results/controller_comparison_summary.csv`
  - `results/controller_mechanics_summary.csv`
- Visualization:
  - `visualization/plots/normalized_reward_vs_episodes.png`
  - `visualization/plots/physical_energy_comparison.png`
  - `visualization/plots/physical_power_comparison.png`
- Paper draft:
  - `paper/document`

---

## 9) Next Validation Targets

- Retrain with latest reward/config and re-run full baseline evaluation
- Target outcomes:
  - RL energy >= PID
  - RL yaw movement < PID
  - RL oscillation substantially reduced vs previous RL checkpoint

---

## 10) Update Timeline

### Phase A - Foundation Complete
- Implemented wind, turbine, and power simulation modules.
- Added dual-mode power equations (`normalized`, `physical`).
- Built Gym-style RL environment with 5D normalized state and 3 discrete actions.

### Phase B - Training and Evaluation Stack
- Added Q-learning baseline and DQN trainer.
- Added model checkpoint + training logs outputs (`rl/model.pth`, `rl/logs.csv`).
- Added controller comparison pipeline (Rule/PID/RL) and statistics (`results/controller_comparison_runs.csv`, `results/controller_comparison_summary.csv`).

### Phase C - Visualization and Reporting
- Added normalized and physical plots under `visualization/plots`.
- Added Streamlit dashboard (`dashboard/app.py`) with robust path inputs.
- Added paper draft (`paper/document`) with all required sections.

### Phase D - Reward/Training Optimization Updates (Most Recent)
- Updated reward shaping in `env/environment.py` to:
  - increase power emphasis (`+1.50 * power_normalized`)
  - stronger alignment shaping (`+0.80 * alignment_term`)
  - explicit misalignment penalty (`-0.70 * normalized_error^2`)
  - movement penalty (`-0.06 * abs(yaw_change_applied)`)
  - direction-switch penalty (`-0.30 * switched_direction`) + rapid flip penalty (`0.05`)
  - alignment bonus (`+0.15` for `|misalignment| <= 5 deg`)
- Updated DQN training constants in `rl/train.py`:
  - episodes `2200`
  - learning rate `3e-4`
  - epsilon `1.0 -> 0.05` with decay cap at `1500`
  - replay buffer `50000`, batch size `256`
  - train start `2000`, target update `250`, grad clip `5.0`

### Phase E - Current Outcome Snapshot
- Configuration updates are active and verified from code and logs.
- Latest run shows stable training convergence, but RL still trails PID in energy during controller comparison evaluation.
