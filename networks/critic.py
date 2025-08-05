import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, input_dim):
        super(Critic, self).__init__()

        self._critic = nn.Linear(input_dim, 1)

    def forward(self, x):
        value = self._critic(x)

        return value
