import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from networks.feed_forward import FeedForward
from networks.actor import Actor
from networks.critic import Critic


class RecurrentPolicy(AbstractPolicy):
    def __init__(self, input_dim, actions_dim, training_config):
        super(RecurrentPolicy, self).__init__()

        self._training_config = training_config

        self._actor_lstm = nn.LSTM(input_dim, 64, 2)
        self._critic_lstm = nn.LSTM(input_dim, 64, 2)

        self._actor_feedforward = FeedForward(64, 4)
        self._critic_feedforward = FeedForward(64, 4)

        self._actor = Actor(6, actions_dim)
        self._critic = Critic(6)

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
        flattened_board = torch.flatten(observation["board"], start_dim=1)

        with torch.no_grad():
            h, _ = self._actor_lstm(flattened_board)
            x = self._actor_feedforward(h)
            normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            action, action_log_probabilities, entropy, _ = self._actor(torch.cat((normalized_x, observation["interoception"]),1))

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        flattened_board = torch.flatten(observation["board"], start_dim=1)

        with torch.no_grad():
            h, _ = self._critic_lstm(flattened_board)
            x = self._critic_feedforward(h)
            normalized_x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            value = self._critic(torch.cat((normalized_x, observation["interoception"]),1))

        return value

    def get_last_actor_h(self):
        return self._last_actor_h

    def get_last_critic_h(self):
        return self._last_critic_h

    def train(self, batch):

        observation = super()._format_observation(batch["board"], batch["interoception"])
        flattened_board = torch.flatten(batch["board"], start_dim=1)

        h_actor, _ = self._actor_lstm(flattened_board)
        x_actor = self._actor_feedforward(h_actor)
        normalized_x_actor = (x_actor - x_actor.mean(dim=1, keepdim=True)) / (x_actor.std(dim=1, keepdim=True) + 1e-8)
        _, _, entropies, distribution = self._actor(torch.cat((normalized_x_actor, batch["interoception"]),1))

        new_action_log_probs = distribution.log_prob(batch["action"])

        h_critic, _ = self._critic_lstm(flattened_board)
        x_critic = self._critic_feedforward(h_critic)
        normalized_x_critic = (x_critic - x_critic.mean(dim=1, keepdim=True)) / (x_critic.std(dim=1, keepdim=True) + 1e-8)
        values = self._critic.forward(torch.cat((normalized_x_critic, batch["interoception"]),1))

        critic_loss = super()._get_critic_loss(values, batch["return"])

        ratios =  super()._get_ratios(
            new_action_log_probs,
            batch["action_log_probabilities"]
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
