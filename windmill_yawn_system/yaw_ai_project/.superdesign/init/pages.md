# Key pages

## `/` (Yaw control dashboard)

Entry: `dashboard/app.py`

Dependencies:
- `dashboard/app.py`
  - `rl/agents.py` — feedforward checkpoint model and hidden-size inference
  - `rl/temporal_agents.py` — LSTM checkpoint model and causal sequence builder
  - `simulation/power_model.py` — normalized and physical power calculations
  - `simulation/turbine_model.py` — yaw simulation
  - `simulation/wind_model.py` — seeded wind trace generation

The entry file contains all UI rendering branches: `render_trace_section`, `render_training_section`, `render_advanced_neural_section`, and `render_comparison_section`.
