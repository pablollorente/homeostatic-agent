import networks as net
import torch

from torch.optim import Adam

class BasicPolicy(Policy):
    def __init__(self, input_dim, actions_dim, training_config: TrainingConfig):
        super(BasicPolicy, self).__init__()

        self._actor_feedforward = net.FeedForward(input_dim, 64)

        self._critic_feedforward = net.FeedForward(input_dim, 64)

        self._actor = net.Actor(64, actions_dim)

        self._actor_optimizer = 0

        self._critic_optimizer = 0

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

    def train(self, replay_buffer, training_config: TrainingConfig):

        critic_loss = F.mse_loss(values, batch['returns'])

        # Calcular ratios para la función de pérdida del actor
        ratios = torch.exp(actions_log_probabilities - batch['actions_log_probabilities'])

        # Calcular la pérdida con clipping y regularización con entropía del actor
        actor_loss = ratios * batch['advantages']
        actor_loss_clipped = torch.clamp(ratios, 1.0 - training_config.get_clip_epsilon(), 1.0 + training_config.get_clip_epsilon()) * batch['advantages']
        actor_loss = -torch.min(actor_loss, actor_loss_clipped).mean()
        actor_loss = actor_loss - training_config.get_entropy_coef * entropies.mean()

        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._actor.parameters(), training_config.get_max_grad_norm())
        self._actor_optimizer.step()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        # TODO nn.utils.clip_grad_norm_(self._critic.parameters(), training_config.get_max_grad_norm())
        self._critic_optimizer.step()

        return

