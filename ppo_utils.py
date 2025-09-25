import torch


class PPOUtils:

    @staticmethod
    def get_returns_and_advantages(rewards, values, dones, gamma = 0.99, gae_lambda = 0.95, device = "cpu"):
        """
         Calcula los retornos y ventajas estandarizadas para PPO usando GAE.

         Returns:
             returns: Tensor de retornos
             advantages: Tensor de ventajas
         """
        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)

        # Inicializar valores para último paso
        next_return = 0
        next_value = 0
        next_advantage = 0

        n_steps = len(rewards)

        # Iterar hacia atrás para calcular retornos y ventajas
        for step in reversed(range(n_steps)):
            # Si es terminal, reset
            if dones[step]:
                next_return = 0
                next_value = 0
                next_advantage = 0

            # Calcular retorno
            returns[step] = rewards[step] + gamma * next_return * (1 - int(dones[step]))

            # Calcular ventaja usando GAE (Generalized Advantage Estimation)
            delta = rewards[step] + gamma * next_value * (1 - int(dones[step])) - values[step]
            advantages[step] = delta + gamma * gae_lambda * next_advantage * (1 - int(dones[step]))

            # Actualizar para próxima iteración
            next_return = returns[step]
            next_value = values[step]
            next_advantage = advantages[step]

        # Estandarizar retornos y ventajas para estabilidad en el entrenamiento
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return normalized_returns, normalized_advantages