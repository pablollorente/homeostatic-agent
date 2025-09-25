import torch.nn as nn


class InteroceptionPredictor(nn.Module):

    def __init__(self, input_dim):
        super(InteroceptionPredictor, self).__init__()

        self._interoception_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        predicted_interoception = self._interoception_predictor(x)

        return predicted_interoception
