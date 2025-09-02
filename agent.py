import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from imagination.generator import Generator
from imagination.discriminator import Discriminator
from conditioning.interoception_predictor import InteroceptionPredictor
from ppo_policies.imagination_policy import ImaginationPolicy
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

        if self._conditioned_responses:
            self._set_interoceptor_predictor()

        if self._imagination:
            self._set_imagination()

    def _set_interoceptor_predictor(self):
        self._interoception_conv = Convolutional().to(self._device)
        self._interoception_feedforward = FeedForward(1024, 4).to(self._device)
        self._interoceptor_predictor = InteroceptionPredictor(6).to(self._device)

        self._intero_optimizer = optim.Adam(
            self._interoceptor_predictor.parameters(),
            lr=self._training_config.get_lr()
        )

    def _set_imagination(self):
        self._generator = Generator().to(self._device)
        self._discriminator = Discriminator().to(self._device)

        self._generator_optimizer = optim.Adam(
            self._generator.parameters(),
            lr=self._training_config.get_lr()
        )

        self._discriminator_optimizer = optim.Adam(
            self._discriminator.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation, info = None):
        # Realizar una acción aleatoria excepto quedarse quieto cuando el monstruo hace daño al agente
        if info and self._innate_responses:
            if info["damage"]:
                self._innate_action = True
                action = torch.tensor(random.randint(1,4), dtype=torch.uint8).unsqueeze(0).to(self._device)
                _, action_log_probs, entropy = self._ppo_policy.get_random_action_pobs_and_entropy()
                return action, action_log_probs, entropy

        # Realizar acción condicionada
        if info and self._conditioned_responses:
            result = self._last_predicted_interoception[0][1] - self.get_interoceptive_state()[1]
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

    def imagine(self, previous_h):
        with torch.no_grad():
            return self._generator(previous_h)

    def get_last_h(self):
        if isinstance(self._ppo_policy, ImaginationPolicy):
            return self._ppo_policy.get_last_h()
        return None

    def train_imagination(self, batch):
        """
        Entrena la GAN (Generator + Discriminator) para el sistema de imaginación.

        Args:
            batch: Datos del batch

        Returns:
            tuple: Tupla con las pérdidas del entrenamiento GAN
        """
        # Datos reales para el discriminador
        real_boards = batch['board']  # Boards reales observados
        real_interoception = batch['interoception']  # Interocepción real
        h = batch['hidden_state']  # Hidden state de las LSTM (solo para el generador)

        # Generar datos imaginados con el generador
        generated_boards, generated_interoception = self._generator(h)

        # Entrenar discriminador
        self._discriminator_optimizer.zero_grad()

        # Predicciones del discriminador para datos reales
        real_predictions = self._discriminator(real_boards, real_interoception)
        real_labels = torch.ones_like(real_predictions)  # Etiquetas "real" = 1

        # Predicciones del discriminador para datos generados (sin hidden state, detach para no entrenar generador)
        fake_predictions = self._discriminator(generated_boards.detach(), generated_interoception.detach())
        fake_labels = torch.zeros_like(fake_predictions)  # Etiquetas "fake" = 0

        # Pérdida del discriminador (Binary Cross Entropy)
        discriminator_loss = F.binary_cross_entropy(real_predictions, real_labels) + \
                             F.binary_cross_entropy(fake_predictions, fake_labels)

        discriminator_loss.backward()
        self._discriminator_optimizer.step()

        # Entrenar generador
        self._generator_optimizer.zero_grad()

        # El generador quiere engañar al discriminador
        # Predicciones del discriminador para datos generados (sin hidden state, sin detach para entrenar generador)
        generator_fake_predictions = self._discriminator(generated_boards, generated_interoception)
        generator_labels = torch.ones_like(
            generator_fake_predictions)  # El generador quiere que sean clasificados como "real"

        # Pérdida del generador
        generator_loss = F.binary_cross_entropy(generator_fake_predictions, generator_labels)

        generator_loss.backward()
        self._generator_optimizer.step()

        return generator_loss.item(), discriminator_loss.item()

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


