# Data collection
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


class Agent:

    def __init__(self, policy, training_config, innate_responses = False, conditioned_responses = False, devide = "cpu"):
        # Detección del dispositivo para el cálculo de los grafos computacionales (CPU/GPU)
        self._device = device

        # Creación de la memoria de reproducción
        self._replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(training_config.get_replay_buffer_size()),
            sampler=SamplerWithoutReplacement(),
            batch_size=training_config.get_batch_size()
        )

        self._ppo = PPO(policy, training_config)

        self._innate_responses = innate_responses
        self._conditioned_responses = conditioned_responses

    def select_action(self, observation):
        return self._ppo.select_action(observation)
