import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from ppo_policies.recurrent_policy_interface import RecurrentPolicyInterface
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from networks.actor import Actor
from networks.critic import Critic


class ConvRecurrentPolicy(AbstractPolicy, RecurrentPolicyInterface):
    def __init__(self, actions_dim, training_config):
        super(ConvRecurrentPolicy, self).__init__()

        self._training_config = training_config

        self._actor_conv = Convolutional()
        self._critic_conv = Convolutional()

        self._actor_lstm = nn.LSTM(1026, 64, 2)
        self._critic_lstm = nn.LSTM(1026, 64, 2)

        self._actor = Actor(64, actions_dim)
        self._critic = Critic(64)

        self._last_actor_h = None
        self._last_critic_h = None

        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._training_config.get_lr()
        )

        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation):
        with(torch.no_grad()):
            x_conv = self._actor_conv(observation["board"])
            h, _ = self._actor_lstm(torch.cat((x_conv, observation["interoception"]), 1))
            action, action_log_probabilities, entropy = self._actor(h)

            self._last_actor_h = h

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        with(torch.no_grad()):
            x_conv = self._critic_conv(observation["board"])
            h, _ = self._critic_lstm(torch.cat((x_conv, observation["interoception"]), 1))
            value = self._critic(h)

            self._last_critic_h = h

        return value

    def get_last_actor_h(self):
        return self._last_actor_h

    def get_last_critic_h(self):
        return self._last_critic_h

    def train(self, batch):
        actor_lstm_features = self._actor_conv(batch["board"])
        h_actor, _ = self._actor_lstm(torch.cat((actor_lstm_features, batch["interoception"]), 1))
        _, _, entropies, distribution = self._actor(h_actor)

        new_action_log_probs = distribution.log_prob(batch["action"])

        critic_lstm_features = self._critic_conv(batch["board"])
        h_critic, _ = self._critic_lstm(torch.cat((critic_lstm_features, batch["interoception"]), 1))
        values = self._critic(h_critic)

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