import argparse
import gymnasium as gym
import torch
import time

from agent import Agent
from survival_env.survival_env import SurvivalEnv
from ppo_policies.basic_policy import BasicPolicy
from training_config import TrainingConfig
from survival_env.env_transformer import EnvTranformer
# Data collection
from tensordict import TensorDict
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


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

    def _run_training(self, agent, replay_buffer, training_config, training_count):
        print("###############################################")
        print(f"# Comenzando el ciclo de entrenamiento {training_count}. #")
        print("###############################################")

        start = time.time()

        for epoch in range(training_config.get_epochs()):
            print(f"Época {epoch + 1}:")
            print("-----------------------------------------------")

            for batch_num in range(training_config.get_n_batches()):
                batch = replay_buffer.sample()

                actor_loss, critic_loss, entropy = agent.train(batch)

                print(
                    f"Batch {batch_num + 1}/{training_config.get_n_batches()}. Actor loss: {actor_loss}, critic loss: {critic_loss}, entropy: {entropy}")

        torch.cuda.synchronize()

        end = time.time()

        training_time = end - start

        training_time = time.strftime("%H horas %M minutos %S segundos", training_time)

        print(f"Duración: {traingin_time}")

        # TODO guardar métricas del entrenamiento

    def run(self):
        #TODO set the rest of the args.

        #TODO crear las clases para guardar las métricas y generar las gráficas.

        env = SurvivalEnv()

        # TODO sacar las dimensiones programáticamente mediante los espacios de la observación del entorno
        policy = BasicPolicy(770, 5)
        training_config = TrainingConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = Agent(policy, training_config, device=device)

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(training_config.get_replay_buffer_size()),
            sampler=SamplerWithoutReplacement(),
            batch_size=training_config.get_batch_size()
        )

        step_count = 0
        training_count = 0

        for episode in range(0, self._args.episodes):
            done = False
            episode_duration = 0
            total_reward = 0

            observation, info = env.reset()

            tensor_observation, tensor_reward = EnvTranformer.to_tensor(observation, device=device)

            while not done:
                print(f"DEBUG -> Interoception {observation['interoception']}")
                print(f"DEBUG -> Interoception {tensor_observation['interoception']}")

                action, action_log_probabilities, entropy = agent.select_action(tensor_observation)

                print(f"DEBUG -> Action {action}")

                value = agent.predict_value(tensor_observation)

                print(f"DEBUG -> Value {value}")

                observation, reward, terminated, truncated, info = env.step(action.item())

                print(f"DEBUG -> Reward {reward}")
                print(f"DEBUG -> Terminated {terminated}")
                print(f"DEBUG -> Truncated {truncated}")
                print(f"DEBUG -> Info {info}")

                tensor_observation, tensor_reward = EnvTranformer.to_tensor(observation, reward, device)

                done = terminated or truncated

                step_data = TensorDict(
                    {
                        "board": tensor_observation["board"].squeeze(),
                        "interoception": tensor_observation["interoception"].squeeze(),
                        "action": action.squeeze(),
                        "action_log_probabilities": action_log_probabilities.squeeze(),
                        "entropy": entropy.squeeze(),
                        "value": value.squeeze(),
                        "reward": tensor_reward.squeeze(),
                        "done": torch.tensor(done, device=device)
                    }
                )

                replay_buffer.add(step_data)

                if len(replay_buffer) >= training_config.get_min_buffer_size() and step_count % training_config.get_train_per_steps() == 0 and not done:
                    training_count += 1
                    self._run_training(agent, replay_buffer, training_count)

                step_count += 1
                episode_duration += 1
                total_reward += reward

            print("################################")
            print(f"# Resumen del episodio {episode + 1}. #")
            print("################################")
            print(f"- Duración: {episode_duration} steps.")
            print(f"- Recompensa acumulada: {total_reward: .2f}.")
            print(f"- Recompensa media: {total_reward / episode_duration:.2f}")

        # TODO guardar métricas del episodio

        env.close()
