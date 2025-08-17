import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FeedForward, self).__init__()

        self._feedforward = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self._feedforward(x)
