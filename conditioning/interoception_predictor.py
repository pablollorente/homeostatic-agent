import torch.nn as nn


class InteroceptionPredictor(nn.Module):

    def __init__(self, input_dim):
        super(InteroceptionPredictor, self).__init__()

        self._interoception_predictor = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        predicted_interoception = self._interoception_predictor(x)

        return predicted_interoception
