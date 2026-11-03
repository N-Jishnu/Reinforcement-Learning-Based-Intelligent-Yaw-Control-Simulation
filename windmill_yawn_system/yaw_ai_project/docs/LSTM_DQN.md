# LSTM-DQN and temporal attention

`rl/temporal_agents.py` adds a separate LSTM-DQN implementation; the existing
feedforward DQN and Double DQN are unchanged.  A causal sequence holds 20
observations by default, each with the existing five yaw-control state
features.  At an episode start, the first observation is repeated as left
padding; previous-episode data is discarded.  Replay stores the complete
state/next-state sequence pair, action, reward, and terminal flag, so sampled
sequences preserve chronology and cannot cross an episode boundary.

The LSTM uses an input gate to decide which new wind/yaw information enters the
cell state, a forget gate to discard irrelevant history, and an output gate to
expose part of the cell state as the hidden representation.  The final hidden
representation is mapped through a linear head to three Q-values.  Terminal
transitions use `r + gamma * Q(next) * (1 - done)`, so they never bootstrap.

The optional Attention-LSTM-DQN computes a score for each LSTM hidden state,
normalizes scores with softmax across time, and forms a weighted sum before the
linear Q head.  During evaluation it writes one row of attention weights per
decision; these are evidence of where the model attended, not evidence that
attention improved control performance.
