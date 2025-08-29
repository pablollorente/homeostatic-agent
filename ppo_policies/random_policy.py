import torch

from ppo_policies.abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, actions_dim):
        super(RandomPolicy, self).__init__()

        self._actions_dim = actions_dim

    def select_action(self, observation):
        return super().get_random_action_pobs_and_entropy()

    def predict_value(self, observation):
        return torch.tensor(0)

    def train(self, batch):
        return torch.tensor(0), torch.tensor(0), torch.tensor(0)