import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()

        self._feedforward = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, output_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self._feedforward(x)
