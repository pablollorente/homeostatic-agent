import random
import torch
import torch.nn.functional as F

from torch.distributions import Categorical
from ppo_policies.abstract_policy import AbstractPolicy


class RandomPolicy(AbstractPolicy):
    def __init__(self, actions_dim):
        super(RandomPolicy, self).__init__()

        self._actions_dim = actions_dim

    def select_action(self, observation):
        actions_logits = torch.randint(0, 100, (1, 5), dtype=torch.float32)
        actions_probabilities = F.softmax(actions_logits, 1)
        distribution = Categorical(actions_probabilities)
        action = distribution.sample()
        action_log_probabilities = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        return torch.tensor(0)

    def train(self, batch):
        return torch.tensor(0), torch.tensor(0), torch.tensor(0)