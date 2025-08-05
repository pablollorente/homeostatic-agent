import torch
import torch.nn as nn


class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self._conv(x)
