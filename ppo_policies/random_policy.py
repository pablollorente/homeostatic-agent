import random
import networks as net
import torch

class RandomPolicy(Policy):
    def __init__(self, input_dim, actions_dim):
        super(RandomPolicy, self).__init__()

        self._actions_dim = actions_dim

    def select_action(self, observation):
        return random.randint(0, self._actions_dim - 1)

    def predict_value(self, observation):
        return 0

    def train(self):
        return