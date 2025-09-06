import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from ppo_policies.imagination_policy import ImaginationPolicy
from networks.feed_forward import FeedForward
from networks.convolutional import Convolutional
from networks.actor import Actor
from networks.critic import Critic


class ConvRecurrentPolicy(AbstractPolicy, ImaginationPolicy):
    def __init__(self, actions_dim, training_config, imagination = False, device = "cpu"):
        super().__init__(device)

        self._training_config = training_config

        self._imagination = imagination

        self._actor_board_processor = Convolutional().to(device)
        self._critic_board_processor = Convolutional().to(device)

        self._actor_feedforward = FeedForward(1024, 6).to(device)
        self._critic_feedforward = FeedForward(1024, 6).to(device)

        lstm_input_dim = 8

        if self._imagination:
            self._actor_imagined_board_processor = Convolutional().to(device)
            self._critic_imagined_board_processor = Convolutional().to(device)

            self._actor_imagination_feedforward = FeedForward(1024, 6).to(device)
            self._critic_imagination_feedforward = FeedForward(1024, 6).to(device)

            lstm_input_dim = 16

        self._actor_lstm = nn.LSTM(lstm_input_dim, 64, 2).to(device)
        self._critic_lstm = nn.LSTM(lstm_input_dim, 64, 2).to(device)

        self._actor = Actor(64, actions_dim).to(device)
        self._critic = Critic(64).to(device)

        self._last_actor_h = torch.zeros(64, dtype=torch.float).unsqueeze(0).to(device)
        self._last_critic_h = torch.zeros(64, dtype=torch.float).unsqueeze(0).to(device)

        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._training_config.get_lr()
        )

        self._critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=self._training_config.get_lr()
        )

    def select_action(self, observation):
        with torch.no_grad():
            x_conv = self._actor_board_processor(observation["board"])
            x = self._actor_feedforward(x_conv)
            lstm_features = torch.cat((x, observation["interoception"]), dim=1)

            if self._imagination:
                x_conv_imagination = self._actor_imagined_board_processor(observation["imagined_board"])
                x_imagination = self._actor_imagination_feedforward(x_conv_imagination)
                lstm_features = torch.cat((lstm_features, x_imagination, observation["imagined_interoception"]), dim=1)

            h, _ = self._actor_lstm(lstm_features)
            action, action_log_probabilities, entropy, _ = self._actor(h)

            self._last_actor_h = h

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        with torch.no_grad():
            x_conv = self._critic_board_processor(observation["board"])
            x = self._critic_feedforward(x_conv)
            lstm_features = torch.cat((x, observation["interoception"]), 1)

            if self._imagination:
                x_conv_imagination = self._critic_imagined_board_processor(observation["imagined_board"])
                x_imagination = self._critic_imagination_feedforward(x_conv_imagination)
                lstm_features = torch.cat((lstm_features, x_imagination, observation["imagined_interoception"]), dim=1)

            h, _ = self._critic_lstm(lstm_features)

            value = self._critic(h)

            self._last_critic_h = h

        return value

    def get_last_h(self):
        return torch.cat((self._last_actor_h, self._last_critic_h), dim=1)

    def train(self, batch):
        actor_feedforward_features = self._actor_board_processor(batch["board"])
        x_actor = self._actor_feedforward(actor_feedforward_features)
        actor_lstm_features = torch.cat((x_actor, batch["interoception"]), dim=1)

        if self._imagination:
            actor_imagination_feedforward_features = self._actor_imagined_board_processor(batch["imagined_board"])
            x_imagination_actor = self._actor_imagination_feedforward(actor_imagination_feedforward_features)
            actor_lstm_features = torch.cat((actor_lstm_features, x_imagination_actor, batch["imagined_interoception"]), dim=1)

        h_actor, _ = self._actor_lstm(actor_lstm_features)
        _, _, entropies, distribution = self._actor(h_actor)

        new_action_log_probs = distribution.log_prob(batch["action"])

        critic_feedforward_features = self._critic_board_processor(batch["board"])
        x_critic = self._critic_feedforward(critic_feedforward_features)
        critic_lstm_features = torch.cat((x_critic, batch["interoception"]), dim=1)

        if self._imagination:
            critic_imagination_feedforward_features = self._critic_imagined_board_processor(batch["imagined_board"])
            x_imagination_critic = self._critic_imagination_feedforward(critic_imagination_feedforward_features)
            critic_lstm_features = torch.cat((critic_lstm_features, x_imagination_critic, batch["imagined_interoception"]), dim=1)

        h_critic, _ = self._critic_lstm(critic_lstm_features)
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