class PPO:
    def __init__(self, policy, training_config):
        self._policy = policy
        self._training_config = training_config

    def get_returns_and_advantages(self, rewards, values, dones):
        """
         Calcula los retornos y ventajas estandarizadas para PPO usando GAE.

         Returns:
             returns: Tensor de retornos
             advantages: Tensor de ventajas
         """
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        gamma = self._training_config.get_gamma()
        gae_lambda = self._training_config.get_gae_lambda()

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
            returns[step] = rewards[step] + gamma * next_return * (1 - dones[step])

            # Calcular ventaja usando GAE (Generalized Advantage Estimation)
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            advantages[step] = delta + gamma * gae_lambda * next_advantage * (1 - dones[step])

            # Actualizar para próxima iteración
            next_return = returns[step]
            next_value = values[step]
            next_advantage = advantages[step]

        # Estandarizar retornos y ventajas para estabilidad en el entrenamiento
        normalized_returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return normalized_returns, normalized_advantages

    def select_action(self, observation):
        self._policy.select_action(observation)
        return

    def predict_value(self, observation):
        return self._policy.predict_value(observation)

    def train(self, batch):
        return self._policy.train(batch, self._training_config)