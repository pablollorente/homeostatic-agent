import random
import torch
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from conditioning.interoception_predictor import InteroceptionPredictor


class Agent:

    def __init__(
            self,
            ppo_policy,
            training_config,
            innate_responses = False,
            conditioned_responses = False,
            imagination = False,
            device = "cpu"
    ):
        # Detección del dispositivo para el cálculo de los grafos computacionales (CPU/GPU)
        self._device = device

        self._ppo_policy = ppo_policy

        self._trainin_config = training_config

        self._innate_responses = innate_responses
        self._conditioned_responses = conditioned_responses

        self._innate_action = False

        self._last_predicted_interoception = None

        self._set_interoceptor_predictor()

    def _set_interoceptor_predictor(self):
        self._interoception_conv = Convolutional()
        self._interoceptor_predictor = InteroceptionPredictor(1026)

    def select_action(self, observation, info = None):
        # Realizar una acción aleatoria excepto quedarse quieto cuando el monstruo hace dao al agente
        if info and self._innate_responses:
            if info["damage"]:
                self._innate_action = True
                return torch.tensor(random.randint(1,4), dtype=torch.uint8).unsqueeze(0)

        return self._ppo_policy.select_action(observation)

    def predict_value(self, observation):
        return self._ppo_policy.predict_value(observation)

    def train(self, batch):
        return self._ppo_policy.train(batch)

    def get_last_predicted_interoception(self):
        return self._last_predicted_interoception

    def predict_interoception(self, observation):
        with(torch.no_grad()):
            x_conv = self._interoception_conv(observation["board"])
            x = self._interoceptor_predictor(torch.cat((x_conv, observation["interoception"]),1))
            prediction = self._interoceptor_predictor(x)

            self._last_predicted_interoception = prediction.item()

        return prediction

    def train_interoception_prediction(self, batch):
        x_conv = self._interoception_conv(batch["board"])
        prediction = self._interoceptor_predictor(torch.cat((x_conv, batch["interoception"]), 1))

        loss = F.mse_loss(prediction.view(-1), batch["next_interoception"])

        optimizer = optim.Adam(
            self._interoceptor_predictor.parameters(),
            lr=self._training_config.get_lr()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def is_last_action_innate(self):
        return self._innate_action
