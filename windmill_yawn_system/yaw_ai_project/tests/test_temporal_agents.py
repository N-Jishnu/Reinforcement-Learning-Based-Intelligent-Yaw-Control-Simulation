import numpy as np
import torch

from rl.temporal_agents import AttentionLSTMQNetwork, LSTMDQNAgent, LSTMQNetwork, SequenceBuilder, SequenceReplayBuffer, TemporalDQNConfig
from rl.nn_comparisons import ComparisonDQNAgent
from rl.agents import DQNConfig

def test_lstm_tensor_dimensions():
    model = LSTMQNetwork(input_size=5, hidden_size=16, action_size=3)
    assert model(torch.zeros(4, 20, 5)).shape == (4, 3)

def test_sequence_builder_resets_and_never_crosses_episodes():
    builder = SequenceBuilder(3)
    first = builder.reset(np.zeros(5)); builder.append(np.ones(5)); second = builder.current()
    third = builder.reset(np.full(5, 7.0))
    assert np.all(first == 0) and np.all(second[-1] == 1) and np.all(third == 7)

def test_sequence_replay_preserves_shapes_and_terminal_flag():
    replay = SequenceReplayBuffer(5); state = np.zeros((3, 5), dtype=np.float32)
    replay.add(state, 1, 2.0, state + 1, True)
    sequences, actions, rewards, next_sequences, dones = replay.sample(1)
    assert sequences.shape == next_sequences.shape == (1, 3, 5)
    assert actions[0] == 1 and rewards[0] == 2.0 and dones[0]

def test_terminal_target_has_no_bootstrap():
    agent = LSTMDQNAgent(TemporalDQNConfig(hidden_size=8, device="cpu"))
    rewards = torch.tensor([[3.0]]); next_states = torch.zeros(1, 20, 5); dones = torch.ones(1, 1)
    assert torch.allclose(agent._target(rewards, next_states, dones), rewards)

def test_attention_weights_are_normalized():
    model = AttentionLSTMQNetwork(input_size=5, hidden_size=8, action_size=3)
    q, weights = model(torch.zeros(2, 20, 5), return_attention=True)
    assert q.shape == (2, 3) and weights.shape == (2, 20)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)

def test_temporal_checkpoint_round_trip(tmp_path):
    agent = LSTMDQNAgent(TemporalDQNConfig(hidden_size=8, device="cpu")); path = tmp_path / "lstm.pth"; agent.save(path)
    loaded = LSTMDQNAgent.load(path, device="cpu")
    assert loaded.config.hidden_size == 8

def test_configurable_feedforward_variants_keep_linear_q_output():
    for activation in ("relu", "leaky_relu", "tanh"):
        agent = ComparisonDQNAgent(DQNConfig(device="cpu"), activation=activation)
        assert agent.online(torch.zeros(2, 5)).shape == (2, 3)
