import argparse
import ale_py
import gymnasium as gym
import torch

from agent import Agent
from survival_env.survival_env import SurvivalEnv
from survival_env.to_tensor import ObservationTranformer
from ppo_policies.basic_policy import BasicPolicy
from training_config import TrainingConfig
from survival_env.env_transformer import EnvTranformer


class App:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Ejecutar el agente homeostático con imaginación en el entorno de supervivencia"
        )
        parser.add_argument(
            "--episodes",
            type=int,
            default=100,
            help="Número de episodios"
        )
        parser.add_argument(
            "--render",
            action="store_true",
            help="Renderizar el entorno durante la ejecución"
        )
        parser.add_argument(
            "--model-path",
            type=str,
            help="Ruta al modelo guardado que se quiere ejecutar"
        )
        parser.add_argument(
            "--save-path",
            type=str,
            default="models",
            help="Ruta para guardar modelos"
        )

        self._args = parser.parse_args()

    def run(self):
        gym.register_envs(ale_py)

        env = SurvivalEnv()

        policy = BasicPolicy(256, 5)
        training_config = TrainingConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = Agent(policy, training_config, device=device)

        #TODO set the rest of the args.

        #TODO crear las clases para guardar las métricas y generar las gráficas.

        episode_over = False
        episode_duration = 0
        total_reward = 0

        for episode in range(0, self._args.episodes):
            observation, info = env.reset()

            tensor_observation, tensor_reward = EnvTranformer.to_tensor(observation, device=self._device)

            while not done:
                action = agent.select_action(tensor_observation)

                observation, reward, terminated, truncated, info = env.step(action)

                tensor_observation, tensor_reward = EnvTranformer.to_tensor(observation, reward, self._device)

                total_reward += reward

                #TODO hacer aquí el entrenamiento

                done = terminated or truncated

        episode_duration += 1

        env.close()
