import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from conditioning.interoception_predictor import InteroceptionPredictor
from survival_env.homeostatic_agent import HomeostaticAgent


class Agent(HomeostaticAgent):

    def __init__(
            self,
            initial_state,
            ppo_policy,
            training_config,
            innate_responses = False,
            conditioned_responses = False,
            imagination = False,
            device = "cpu"
    ):
        super().__init__(np.array(initial_state))

        self._ppo_policy = ppo_policy
        self._training_config = training_config
        self._innate_responses = innate_responses
        self._conditioned_responses = conditioned_responses
        self._imagination = imagination
        self._device = device

        self._last_action = None
        self._innate_action = False
        self._last_predicted_interoception = None

        self._set_interoceptor_predictor()

    def _set_interoceptor_predictor(self):
        self._interoception_conv = Convolutional()
        self._interoceptor_predictor = InteroceptionPredictor(1026)

        self._intero_optimizer = optim.Adam(
            self._interoceptor_predictor.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation, info = None):
        # Realizar una acción aleatoria excepto quedarse quieto cuando el monstruo hace daño al agente
        if info and self._innate_responses:
            if info["damage"]:
                self._innate_action = True
                return torch.tensor(random.randint(1,4), dtype=torch.uint8).unsqueeze(0)

        action, action_log_probabilities, entropy = self._ppo_policy.select_action(observation)

        self._last_action = action.item()

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        return self._ppo_policy.predict_value(observation)

    def train(self, batch):
        return self._ppo_policy.train(batch)

    def get_last_predicted_interoception(self):
        return self._last_predicted_interoception

    def predict_interoception(self, observation):
        with torch.no_grad():
            x_conv = self._interoception_conv(observation["board"])
            normalized_x_conv = (x_conv - x_conv.mean(dim=1, keepdim=True)) / (x_conv.std(dim=1, keepdim=True) + 1e-8)
            prediction = self._interoceptor_predictor(torch.cat((normalized_x_conv, observation["interoception"]),1))

            self._last_predicted_interoception = prediction.detach().clone()

        return prediction

    def train_interoception_prediction(self, batch):
        x_conv = self._interoception_conv(batch["board"])
        normalized_x_conv = (x_conv - x_conv.mean(dim=1, keepdim=True)) / (x_conv.std(dim=1, keepdim=True) + 1e-8)
        prediction = self._interoceptor_predictor(torch.cat((normalized_x_conv, batch["interoception"]), 1))

        loss = F.mse_loss(prediction, batch["next_interoception"])

        self._intero_optimizer.zero_grad()
        loss.backward()
        self._intero_optimizer.step()

        return loss.item()

    def is_last_action_innate(self):
        return self._innate_action

    def update_interoception(self, info):
        """
        Actualiza el estado interoceptivo basado en la observación.

        Args:
            info: Diccionario con la observación del entorno
                'damage': Booleano indicando si el agente recibió daño
                'food': Booleano indicando si el agente encontró comida
                'medicine': Booleano indicando si el agente encontró medicina
        """
        # Disminuir energía en cada paso (coste metabólico) y ejercitar el cuerpo
        if self._last_action == 0:
            self.add_energy(-0.001)
        else:
            self.add_energy(-0.05)
            self.add_integrity(0.01)

        # Consumir comida (aumenta energía)
        if info['food']:
            self.add_energy(0.5)

        # Usar medicina (restaura integridad)
        if info['medicine']:
            self.set_integrity_to_max()

        # Recibir daño del monstruo
        if info['damage']:
            self.add_integrity(-0.1)
