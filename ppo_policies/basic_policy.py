import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from networks.feed_forward import FeedForward
from networks.actor import Actor
from networks.critic import Critic


class BasicPolicy(AbstractPolicy):
    def __init__(self, input_dim, actions_dim, training_config):
        super(BasicPolicy, self).__init__()

        self._training_config = training_config

        self._actor_board_processor = FeedForward(input_dim, 4)
        self._critic_board_processor = FeedForward(input_dim, 4)

        self._actor = Actor(6, actions_dim)
        self._critic = Critic(6)

        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._training_config.get_lr()
        )

        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation):
        board, intero = super()._format_observation_new(observation["board"], observation["interoception"])

        with torch.no_grad():
            board_features = self._actor_board_processor(board)
            # normalized_board_features = (board_features - board_features.min()) / (board_features - board_features.max() + 1e-8)
            action, action_log_probabilities, entropy, _ = self._actor(torch.cat((board_features, intero), 1))

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        board, intero = super()._format_observation_new(observation["board"], observation["interoception"])

        with torch.no_grad():
            board_features = self._critic_board_processor(board)
            # normalized_board_features = (board_features - board_features.min()) / (board_features - board_features.max() + 1e-8)
            value = self._critic(torch.cat((board_features, intero), 1))

        return value

    def train(self, batch):

        board, intero = super()._format_observation_new(batch["board"], batch["interoception"])

        actor_board_features = self._actor_board_processor(board)
        # normalized_actor_board_features = (actor_board_features - actor_board_features.min()) / (actor_board_features - actor_board_features.max() + 1e-8)
        _, _, entropies, distribution = self._actor(torch.cat((actor_board_features, intero), 1))
        new_action_log_probs = distribution.log_prob(batch['action'])

        critic_board_features = self._critic_board_processor(board)
        # normalized_critic_board_features = (critic_board_features - critic_board_features.min()) / (critic_board_features - critic_board_features.max() + 1e-8)
        values = self._critic(torch.cat((critic_board_features, intero), 1))

        critic_loss = super()._get_critic_loss(values, batch["return"])

        ratios =  super()._get_ratios(
            new_action_log_probs,
            batch['action_log_probabilities']
        )

        actor_loss = super()._get_actor_loss(
            ratios,
            batch['advantage'],
            self._training_config.get_clipping_eps(),
            self._training_config.get_entropy_coef(),
            entropies
        )

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self._training_config.get_max_grad_norm())
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self._critic.parameters(), self._training_config.get_max_grad_norm())
        self._critic_optimizer.step()

        return actor_loss, critic_loss, entropies.mean()

