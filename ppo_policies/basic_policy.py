import networks as net
import torch.optim as optim
import torch

from ppo_policies.abstract_policy import AbstractPolicy
from networks.feed_forward import FeedForward
from networks.actor import Actor
from networks.critic import Critic


class BasicPolicy(AbstractPolicy):
    def __init__(self, input_dim, actions_dim):
        super(BasicPolicy, self).__init__()

        self._actor_feedforward = FeedForward(input_dim, 64)

        self._critic_feedforward = FeedForward(input_dim, 64)

        self._actor = Actor(64, actions_dim)

        self._critic = Critic(64)

    def _format_observation(self, observation):
        flattened_board = observation["board"].detach().clone()
        flattened_board = torch.flatten(flattened_board, start_dim=1)

        return torch.cat((flattened_board, observation["interoception"].detach().clone()),1)

    def select_action(self, observation):
        input = self._format_observation(observation)

        with(torch.no_grad()):
            x = self._actor_feedforward(input)
            action, action_log_probabilities, entropy = self._actor(x)

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        input = self._format_observation(observation)

        with(torch.no_grad()):
            x = self._critic_feedforward(input)
            value = self._critic(x)

        return value

    def train(self, batch, training_config):

        actions, actions_log_probabilities, entropies = self._actor.forward(
            batch['observation']
        )

        values = self._critic.forward(batch['observation'])

        critic_loss = super()._get_critic_loss(values, returns)

        ratios =  super()._get_ratios(
            actions_log_probabilities,
            batch['actions_log_probabilities']
        )

        actor_loss = super()._get_actor_loss(ratios, batch['advantages'], training_config.get_clipping_eps())

        actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=training_config.get_lr()
        )

        critic_optimizer = optim.Adam(
            self._critic.parameters(),
            lr=training_config.get_lr()
        )

        actor_optimizer.zero_grad()
        actor_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._actor.parameters(), training_config.get_max_grad_norm())
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._critic.parameters(), training_config.get_max_grad_norm())
        critic_optimizer.step()

        return actor_loss, critic_loss, entropies.mean()

