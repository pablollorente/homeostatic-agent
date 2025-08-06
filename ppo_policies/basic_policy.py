import networks as net
import torch

from networks import actor as Actor, critic as Critic, feed_forward as FeedForward


class BasicPolicy(Policy):
    def __init__(self, input_dim, actions_dim):
        super(BasicPolicy, self).__init__()

        self._actor_feedforward = FeedForward(input_dim, 64)

        self._critic_feedforward = FeedForward(input_dim, 64)

        self._actor = Actor(64, actions_dim)

        self._critic = Critic(input_dim)

    def select_action(self, observation):
        with(torch.no_grad()):
            x = self._actor_feedforward(observation)
            action, action_log_probabilities, entropy = self._actor(x)

        return action, action_log_probabilities, entropy

    def predict_value(self, observation):
        with(torch.no_grad()):
            x = self._critic_feedforward(observation)
            value = self._critic(x)

        return value

    def train(self, batch, training_config: TrainingConfig):

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

        actor_optimizer = training_config.get_actor_optimizer()
        critic_optimizer = training_config.get_critic_optimizer()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._actor.parameters(), training_config.get_max_grad_norm())
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._critic.parameters(), training_config.get_max_grad_norm())
        critic_optimizer.step()

        return actor_loss, critic_loss, entropies.mean()

