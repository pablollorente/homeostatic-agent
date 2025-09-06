import networks as net
import torch.optim as optim
import torch.nn as nn
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from ppo_policies.imagination_policy import ImaginationPolicy
from networks.feed_forward import FeedForward
from networks.actor import Actor
from networks.critic import Critic


class RecurrentPolicy(AbstractPolicy, ImaginationPolicy):
    def __init__(self, input_dim, actions_dim, training_config, imagination = False, device = "cpu"):
        super().__init__(device)

        self._training_config = training_config

        self._imagination = imagination

        self._actor_board_processor = FeedForward(input_dim, 6).to(device)
        self._critic_board_processor = FeedForward(input_dim, 6).to(device)

        lstm_input_dim = 8

        if self._imagination:
            self._actor_imagined_board_processor = FeedForward(input_dim, 6).to(device)
            self._critic_imagined_board_processor = FeedForward(input_dim, 6).to(device)

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
        flattened_board = torch.flatten(observation["board"], start_dim=1)

        with torch.no_grad():
            board_features = self._actor_board_processor(flattened_board)
            #normalized_board_features = (board_features - board_features.min()) / (board_features - board_features.max() + 1e-8)
            lstm_features = torch.cat((board_features, observation["interoception"]), 1)

            if self._imagination:
                flattened_imagined_board = torch.flatten(observation["imagined_board"], start_dim=1)
                imagined_board_features = self._actor_imagined_board_processor(flattened_imagined_board)
                #normalized_imagined_board_features = (imagined_board_features - imagined_board_features.min()) / (imagined_board_features - imagined_board_features.max() + 1e-8)
                lstm_features = torch.cat((lstm_features, imagined_board_features, observation["imagined_interoception"]), 1)

            h, _ = self._actor_lstm(lstm_features)

            self._last_actor_h = h.detach().clone()

            action, action_log_probabilities, entropy, _ = self._actor(h)

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        flattened_board = torch.flatten(observation["board"], start_dim=1)

        with torch.no_grad():
            board_features = self._critic_board_processor(flattened_board)
            #normalized_board_features = (board_features - board_features.min()) / (board_features - board_features.max() + 1e-8)
            lstm_features = torch.cat((board_features, observation["interoception"]), 1)

            if self._imagination:
                flattened_imagined_board = torch.flatten(observation["imagined_board"], start_dim=1)
                imagined_board_features = self._critic_imagined_board_processor(flattened_imagined_board)
                #normalized_imagined_board_features = (imagined_board_features - imagined_board_features.min()) / (imagined_board_features - imagined_board_features.max() + 1e-8)
                lstm_features = torch.cat((lstm_features, imagined_board_features,
                                           observation["imagined_interoception"]), 1)

            h, _ = self._critic_lstm(lstm_features)

            self._last_critic_h = h.detach().clone()

            value = self._critic(h)

        return value

    def get_last_h(self):
        return torch.cat((self._last_actor_h, self._last_critic_h), dim=1)

    def train(self, batch):
        flattened_board = torch.flatten(batch["board"], start_dim=1)

        actor_board_features = self._actor_board_processor(flattened_board)
        actor_lstm_features = torch.cat((actor_board_features, batch["interoception"]), dim=1)

        critic_board_features = self._critic_board_processor(flattened_board)
        critic_lstm_features = torch.cat((critic_board_features, batch["interoception"]), dim=1)

        if self._imagination:
            flattened_imagined_board = torch.flatten(batch["imagined_board"], start_dim=1)

            actor_imagined_board_features = self._actor_imagined_board_processor(flattened_imagined_board)
            actor_lstm_features = torch.cat((actor_lstm_features, actor_imagined_board_features,
                                       batch["imagined_interoception"]), dim=1)

            critic_imagined_board_features = self._critic_imagined_board_processor(flattened_imagined_board)
            critic_lstm_features = torch.cat((critic_lstm_features, critic_imagined_board_features,
                                       batch["imagined_interoception"]), dim=1)

        h_actor, _ = self._actor_lstm(actor_lstm_features)
        _, _, entropies, distribution = self._actor(h_actor)

        new_action_log_probs = distribution.log_prob(batch["action"])

        h_critic, _ = self._critic_lstm(critic_lstm_features)
        values = self._critic(h_critic)

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
