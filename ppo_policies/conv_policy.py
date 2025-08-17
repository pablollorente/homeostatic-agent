import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from networks.actor import Actor
from networks.critic import Critic


class ConvPolicy(AbstractPolicy):
    def __init__(self, actions_dim, training_config):
        super(ConvPolicy, self).__init__()

        self._training_config = training_config

        self._actor_conv = Convolutional()
        self._critic_conv = Convolutional()

        self._actor_feedforward = FeedForward(1026, 64)
        self._critic_feedforward = FeedForward(1026, 64)

        self._actor = Actor(64, actions_dim)
        self._critic = Critic(64)

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
            x = self._actor_feedforward(torch.cat((x_conv, observation["interoception"]),1))
            action, action_log_probabilities, entropy, _ = self._actor(x)

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        with(torch.no_grad()):
            x_conv = self._critic_conv(observation["board"])
            x = self._critic_feedforward(torch.cat((x_conv, observation["interoception"]), 1))
            value = self._critic(x)

        return value

    def train(self, batch):
        actor_feedforward_features = self._actor_conv(batch["board"])
        actor_features = self._actor_feedforward(torch.cat((actor_feedforward_features, batch["interoception"]), 1))
        _, _, entropies, distribution = self._actor(actor_features)

        new_action_log_probs = distribution.log_prob(batch['action'])


        critic_feedforward_features = self._critic_conv(batch["board"])
        critic_features = self._critic_feedforward(torch.cat((critic_feedforward_features, batch["interoception"]), 1))
        values = self._critic(critic_features)

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
        # TODO nn.utils.clip_grad_norm_(self._actor.parameters(), training_config.get_max_grad_norm())
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._critic.parameters(), training_config.get_max_grad_norm())
        self._critic_optimizer.step()

        return actor_loss, critic_loss, entropies.mean()