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
            homeostatis_config,
            innate_responses = False,
            conditioned_responses = False,
            imagination = False,
            device = "cpu"
    ):
        super().__init__(np.array(initial_state))

        self._ppo_policy = ppo_policy
        self._training_config = training_config
        self._homeostasis_config = homeostatis_config
        self._innate_responses = innate_responses
        self._conditioned_responses = conditioned_responses
        self._imagination = imagination
        self._device = device

        self._last_action = None
        self._innate_action = False
        self._conditioned_action = False
        self._last_predicted_interoception = None

        self._set_interoceptor_predictor()

    def _set_interoceptor_predictor(self):
        self._interoception_conv = Convolutional()
        self._interoception_feedforward = FeedForward(1024, 4)
        self._interoceptor_predictor = InteroceptionPredictor(6)

        self._intero_optimizer = optim.Adam(
            self._interoceptor_predictor.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation, info = None):
        # Realizar una acción aleatoria excepto quedarse quieto cuando el monstruo hace daño al agente
        if info and self._innate_responses:
            if info["damage"]:
                self._innate_action = True
                action = torch.tensor(random.randint(1,4), dtype=torch.uint8).unsqueeze(0)
                _, action_log_probs, entropy = self._ppo_policy.get_random_action_pobs_and_entropy()
                return action, action_log_probs, entropy

        # Realizar acción condicionada
        if info and self._conditioned_responses:
            result = self._last_predicted_interoception[0][1] - self.get_interoceptive_state()[1]
            print(f"DEBUG agent.py -> integrity prediction: {self._last_predicted_interoception[0][1]}")
            print(f"DEBUG agent.py -> current integrity: {self.get_interoceptive_state()[1]}")
            print(f"DEBUG agent.py -> prediction difference: {result}")
            if result <= self._homeostasis_config.monster_damage:
                self._conditioned_action = True
                action = self._get_conditioned_action(info)
                _, action_log_probs, entropy = self._ppo_policy.get_random_action_pobs_and_entropy()
                return action, action_log_probs, entropy

        self._innate_action = False
        self._conditioned_action =  False

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
            x = self._interoception_feedforward(x_conv)
            normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            prediction = self._interoceptor_predictor(torch.cat((normalized_x, observation["interoception"]),1))

            self._last_predicted_interoception = prediction.detach().clone()

        return prediction

    def train_interoception_prediction(self, batch):
        x_conv = self._interoception_conv(batch["board"])
        x = self._interoception_feedforward(x_conv)
        normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        prediction = self._interoceptor_predictor(torch.cat((normalized_x, batch["interoception"]), 1))

        loss = F.mse_loss(prediction, batch["next_interoception"])

        self._intero_optimizer.zero_grad()
        loss.backward()
        self._intero_optimizer.step()

        return loss.item()

    def is_last_action_innate(self):
        return self._innate_action

    def is_last_action_conditioned(self):
        return self._conditioned_action

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
            self.add_energy(self._homeostasis_config.basal_metabolic_cost)
        else:
            self.add_energy(self._homeostasis_config.action_metabolic_cost)
            self.add_integrity(self._homeostasis_config.action_integrity_recovery)

        # Consumir comida (aumenta energía)
        if info['food']:
            self.add_energy(self._homeostasis_config.food_energy_recovery)

        # Usar medicina (restaura integridad)
        if info['medicine']:
            self.add_integrity(self._homeostasis_config.medicine_integrity_recovery)

        # Recibir daño del monstruo
        if info['damage']:
            self.add_integrity(self._homeostasis_config.monster_damage)

    def _get_conditioned_action(self, info):
        m_pos = info["monster_position"]
        a_pos = info["agent_position"]

        #Si el monstruo está por debajo del agente
        if m_pos[0] > a_pos[0]:
            # Si el agente se encuentra en el límite superior del tablero
            if a_pos[0] == 0:
                # Si el agente se encuentra en el límite izquierdo
                if a_pos[1] == 0:
                    return torch.tensor([4], dtype=torch.int8)
                # Si el agente se encuentra en el límite derecho
                if a_pos[1] == 15:
                    return torch.tensor([3], dtype=torch.int8)
                return torch.randint(3, 4, (1, 1), dtype=torch.int8)
            # Ir hacia arriba, izquierda o derecha
            posible_actions = [1, 3, 4]
            random.shuffle(posible_actions)
            return torch.tensor([posible_actions[0]], dtype=torch.int8)
        # Si el monstruo está por encima del agente
        elif m_pos[0] < a_pos[0]:
            # Si el agente se encuentra en el límite inferior del tablero
            if a_pos[0] == 15:
                # Si el agente se encuentra en el límite izquierdo
                if a_pos[1] == 0:
                    return torch.tensor([4], dtype=torch.int8)
                # Si el agente se encuentra en el límite derecho
                if a_pos[1] == 15:
                    return torch.tensor([3], dtype=torch.int8)

                return torch.randint(3, 4, (1, 1), dtype=torch.int8)
            # Ir hacia abajo, izquierda o derecha
            return torch.randint(2, 4, (1, 1), dtype=torch.int8)
        # Si el monstruo está en la misma fila
        else:
            # Ambos están en el límite superior
            if a_pos[0] == 0:
                return torch.tensor([2], dtype=torch.int8)
            # Ambos están en el límite inferior
            if a_pos[0] == 15:
                return torch.tensor([1], dtype=torch.int8)
            # Están en cualquier otra fila
            return torch.randint(1, 2, (1, 1), dtype=torch.int8)


