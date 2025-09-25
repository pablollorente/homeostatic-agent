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
    def __init__(self, actions_dim, training_config, device = "cpu"):
        super().__init__(device)

        self._training_config = training_config

        self._actor_conv = Convolutional().to(device)
        self._critic_conv = Convolutional().to(device)

        self._actor_feedforward = FeedForward(1024, 4).to(device)
        self._critic_feedforward = FeedForward(1024, 4).to(device)

        self._actor = Actor(6, actions_dim).to(device)
        self._critic = Critic(6).to(device)

        self._actor_params = list(self._actor_conv.parameters()) + list(self._actor_conv.parameters()) + list(self._actor_conv.parameters())
        self._critic_params = list(self._critic_conv.parameters()) + list(self._actor_conv.parameters()) + list(self._actor_conv.parameters())

        self._actor_optimizer = optim.Adam(
            self._actor_params,
            lr=self._training_config.get_lr()
        )

        self._critic_optimizer = optim.Adam(
            self._critic_params,
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation):
        with torch.no_grad():
            x_conv = self._actor_conv(observation["board"])
            x = self._actor_feedforward(x_conv)
            normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            action, action_log_probabilities, entropy, _ = self._actor(torch.cat((normalized_x, observation["interoception"]),1))

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        with torch.no_grad():
            x_conv = self._critic_conv(observation["board"])
            x = self._critic_feedforward(x_conv)
            normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            value = self._critic(torch.cat((normalized_x, observation["interoception"]), 1))

        return value

    def train(self, batch):
        actor_feedforward_features = self._actor_conv(batch["board"])
        actor_features = self._actor_feedforward(actor_feedforward_features)
        normalized_actor_features = (actor_features - actor_features.mean(dim=1, keepdim=True)) / (actor_features.std(dim=1, keepdim=True) + 1e-8)
        _, _, entropies, distribution = self._actor(torch.cat((normalized_actor_features, batch["interoception"]), 1))

        new_action_log_probs = distribution.log_prob(batch['action'])


        critic_feedforward_features = self._critic_conv(batch["board"])
        critic_features = self._critic_feedforward(critic_feedforward_features)
        normalized_critic_features = (critic_features - critic_features.mean(dim=1, keepdim=True)) / (critic_features.std(dim=1, keepdim=True) + 1e-8)
        values = self._critic(torch.cat((normalized_critic_features, batch["interoception"]), 1))

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
        nn.utils.clip_grad_norm_(self._actor_params, self._training_config.get_max_grad_norm())
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self._critic_params, self._training_config.get_max_grad_norm())
        self._critic_optimizer.step()

        return actor_loss, critic_loss, entropies.mean()