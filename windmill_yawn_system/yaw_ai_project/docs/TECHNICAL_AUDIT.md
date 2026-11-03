# Technical audit: intelligent yaw-control RL project

Audit completed before the final corrective change on 2 November 2026.

## Existing architecture and functionality

The project retains the original wind-turbine simulation while adding a Gymnasium-compatible reinforcement-learning environment at `env/environment.py`. The environment has a five-value normalized observation (wind direction, wind speed, yaw angle, yaw misalignment, and wind-direction change) and three discrete actions (left, right, hold). Its reward combines normalized aerodynamic power, alignment, yaw-misalignment cost, movement cost, and direction-reversal cost.

The neural controller is an MLP implemented in `rl/agents.py`: `5 -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(3)`. It has 17,667 trainable parameters and uses PyTorch `Linear` defaults (Kaiming-uniform weights and uniform biases), Adam (learning rate `3e-4`), MSE TD loss, gradient clipping, a replay buffer, and a periodically synchronized target network. `DQNAgent` uses the standard max-target update; the separate `DoubleDQNAgent` selects a next action with the online network and evaluates it with the target network. Terminal transitions zero the bootstrap term.

The original tabular Q-learning baseline remains available through `rl/train.py` and `rl/agents.py`; legacy simulation, farm simulation, PID, and rule-based workflows remain in place.

## Audited modules

| Area | Main files | Finding |
|---|---|---|
| Environment and physics | `env/environment.py`, `simulation/` | Seeded wind traces, yaw dynamics, normalized and physical power reporting are implemented. |
| Neural agents | `rl/agents.py` | DQN, Double DQN, replay, checkpoints, architecture introspection, and reproducibility helpers are implemented. |
| Training and metrics | `rl/trainer.py`, `rl/metrics.py`, `rl/train.py` | CSV and optional TensorBoard logging cover reward, moving average, loss, Q-values, epsilon, gradients, length, timing, and optimizer steps. |
| Architecture reporting | `rl/architecture.py` | Prints an exact layer/parameter summary and draws a Matplotlib diagram from the actual model configuration. |
| Plots and dashboard | `rl/visualize_training.py`, `dashboard/app.py` | Reads CSV data and handles missing/incomplete files rather than generating synthetic values. |
| Experiments | `rl/experiments.py` | Ablations and multi-seed DQN/Double-DQN runs use shared profiles, separate evaluation environments, summaries, and approximate 95% intervals. |
| Evaluation | `evaluation/agent_evaluation.py`, `rl/trainer.py` | Compares random, Q-learning, DQN, Double DQN, rule, and PID policies on the same seeded traces. Reports normalized/physical power and yaw stability. |
| Tests | `tests/test_rl_core.py`, `tests/test_experiments.py` | Covers shapes, replay, real optimization metrics, target equations, checkpoints, reproducibility, CSVs, experiments, and short end-to-end training. |

## Gaps and risks found

1. **Fabricated episode aggregates before warm-up:** before any gradient update, `trainer.py` wrote zero loss, Q-value, and gradient-norm aggregates. Zero is not an observed measurement and can distort charts or summaries. This is corrected after this audit by writing `NaN` for unavailable measurements.
2. **Training cost:** the full profile is 2,200 x 1,000 environment steps per run. Five seeds and four ablation variants are computationally expensive on CPU. The documented `reduced` profile is suitable for smoke experiments but should be labelled as such when reporting results.
3. **Confidence intervals:** the helper currently uses a normal approximation (`1.96 * s / sqrt(n)`), not a Student-t critical value. This is acceptable only when described as approximate, especially for five seeds.
4. **Legacy duplication:** `rl/evaluate.py` and `rl/plot_training.py` are older compatibility utilities. The enhanced paths are `evaluation/agent_evaluation.py` and `rl/visualize_training.py`.
5. **Path convention:** CLI defaults are project-root-relative; commands must be run from `yaw_ai_project` (as documented).

## Files modified and added by this completion

- Modified: `rl/trainer.py` to stop logging non-observed zero metrics during the optimizer warm-up period.
- Added: this audit document.

## Recommended execution order

1. Run `pytest -q`.
2. Train short DQN and Double-DQN smoke runs.
3. Render plots from the resulting CSV files.
4. Run the five-seed reduced experiment, then controller evaluation using the trained checkpoints.
5. Use TensorBoard with `tensorboard --logdir results/tensorboard` when a run was started with `--tensorboard-dir results/tensorboard`.

---

## Post-audit update: 3 November 2026

The audit scope was extended without modifying the original DQN agent:

- `rl/temporal_agents.py` now provides sequence-safe LSTM-DQN and
  Attention-LSTM-DQN agents, online/target networks, terminal masking,
  checkpointing, gradient clipping and attention extraction.
- `rl/temporal_trainer.py` and `rl/temporal_experiments.py` provide temporal
  training/evaluation and isolated result directories.
- `rl/nn_comparisons.py` provides controlled Adam/RMSProp/SGD-momentum and
  ReLU/Leaky-ReLU/Tanh studies.
- `evaluation/unified_comparison.py` merges available result summaries without
  rewriting their source CSVs.
- `tests/test_temporal_agents.py` adds tests for LSTM dimensions, sequence
  isolation, terminal masking, normalized attention, checkpoint restoration and
  configurable linear Q outputs.

A complete five-seed advanced result bundle imported from Colab is preserved at
`results/colab_final_advanced_experiment/`. Its results and limitations are
documented in `FINAL_ADVANCED_EXPERIMENTS.md`; it supersedes neither legacy nor
local partial results.
