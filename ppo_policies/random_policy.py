import torch

from ppo_policies.abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, actions_dim, device = "cpu"):
        super().__init__(device)

        self._actions_dim = actions_dim
        self._device = device

    def select_action(self, observation):
        return super().get_random_action_pobs_and_entropy()

    def predict_value(self, observation):
        return torch.tensor(0).to(self._device)

    def train(self, batch):
        return torch.tensor(0).to(self._device), torch.tensor(0).to(self._device), torch.tensor(0).to(self._device)