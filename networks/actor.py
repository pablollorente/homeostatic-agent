import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical


class Actor(nn.Module):

    def __init__(self, input_dim, actions_dim):
        super(Actor, self).__init__()

        self._actor = nn.Linear(input_dim, actions_dim)

    def forward(self, x):
        actions_logits = self._actor(x)

        actions_probabilities = F.softmax(actions_logits, 1)

        # Crear distribución categórica de las probabilidades de las acciones
        distribution = Categorical(actions_probabilities)

        # Seleccionar la acción a tomar obteniendo una muestra de la distribución
        action = distribution.sample()

        # Obtenemos la probabilidad logarítimas de cada acción según la política actual para luego calcular los ratios
        action_log_probabilities = distribution.log_prob(action)

        #Obtenemos la entropía de la distribución para añadir una compensación entre explotación y exploración en la función de pérdida
        entropy = distribution.entropy()

        return action, action_log_probabilities, entropy, distribution
