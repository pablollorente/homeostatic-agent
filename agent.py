# Data collection
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


class Agent:

    def __init__(self, ppo_policy, training_config, innate_responses = False, conditioned_responses = False, device = "cpu"):
        # Detección del dispositivo para el cálculo de los grafos computacionales (CPU/GPU)
        self._device = device

        self._ppo_policy = ppo_policy

        self._training_config = training_config

        self._innate_responses = innate_responses
        self._conditioned_responses = conditioned_responses

    def select_action(self, observation):
        return self._ppo_policy.select_action(observation)

    def predict_value(self, observation):
        return self._ppo_policy.predict_value(observation)

    def train(self, batch):
        return self._ppo_policy.train(batch, self._training_config)
