import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch.distributions import Categorical


class AbstractPolicy(ABC):
    @abstractmethod
    def select_action(self, observation):
        pass

    @abstractmethod
    def predict_value(self, observation):
        pass

    @abstractmethod
    def train(self, batch):
        pass

    def _format_observation(self, board, interoception):
        flattened_board = board.detach().clone()
        flattened_board = torch.flatten(flattened_board, start_dim=1)

        return torch.cat((flattened_board, interoception.detach().clone()),1)

    def _format_observation_new(self, board, interoception):
        flattened_board = board.detach().clone()
        flattened_board = torch.flatten(flattened_board, start_dim=1)

        return flattened_board, interoception.detach().clone()

    def _get_ratios(self, new_actions_log_probabilities, old_actions_log_probabilities):
        return torch.exp(new_actions_log_probabilities - old_actions_log_probabilities.detach())

    def _get_actor_loss(self, ratios, advantages, eps, entropy_coef, entropies):
        """
        Computes clipped actor loss with entropy bonus per batch
        """
        actor_loss = ratios * advantages
        actor_loss_clipped = torch.clamp(ratios, 1.0 - eps, 1.0 + eps) * advantages
        min_actor_loss = -torch.min(actor_loss, actor_loss_clipped)
        final_actor_loss = min_actor_loss - entropy_coef * entropies.mean()

        return final_actor_loss.mean()

    def _get_critic_loss(self, values, returns):
        return F.mse_loss(values.view(-1), returns)

    def get_random_action_pobs_and_entropy(self):
        actions_logits = torch.randint(0, 100, (1, 5), dtype=torch.float32)
        actions_probabilities = F.softmax(actions_logits, 1)
        distribution = Categorical(actions_probabilities)
        action = distribution.sample()
        action_log_probabilities = distribution.log_prob(action)
        entropy = distribution.entropy()

        return action, action_log_probabilities, entropy

