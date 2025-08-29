import argparse
import gymnasium as gym
import torch
import time

from agent import Agent
from survival_env.survival_env import SurvivalEnv
from ppo_policies.basic_policy import BasicPolicy
from ppo_policies.conv_policy import ConvPolicy
from ppo_policies.recurrent_policy import RecurrentPolicy
from ppo_policies.conv_recurrent_policy import ConvRecurrentPolicy
from ppo_policies.random_policy import RandomPolicy
from training_config import TrainingConfig
from survival_env.env_transformer import EnvTranformer
from survival_env.reward_calculator import RewardCalculator
from survival_env.homeostasis_config import HomeostasisConfig
from ppo_utils import PPOUtils
from experiment_logging.experiment_logger import ExperimentLogger
from experiment_logging.experiment_plotter import ExperimentPlotter
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
            '--policy',
            choices=['basic', 'conv', 'recurrent', 'convrec', 'random'],
            default="basic",
            help="Tipo de red para la política del agente: básica o feedforward, convolucional, recurrente o convolucional y recurrente."
        )
        parser.add_argument(
            '--intero',
            action="store_true",
            help="El agente utiliza la predicción de la interocepción para generar respuestas condicionadas."
        )
        parser.add_argument(
            '--innate',
            action="store_true",
            help="El agente genera respuestas innatas."
        )
        parser.add_argument(
            '--imagination',
            action="store_true",
            help="El agente genera imagenes del tablero y estados interoceptivos imaginados que alimentan la red su política."
        )
        parser.add_argument(
            "--episodes",
            type=int,
            default=100,
            help="Número de episodios."
        )
        parser.add_argument(
            "--render",
            action="store_true",
            help="Renderizar el entorno durante la ejecución."
        )
        parser.add_argument(
            "--model-path",
            type=str,
            help="Ruta al modelo guardado que se quiere ejecutar."
        )
        parser.add_argument(
            "--save-path",
            type=str,
            default="models",
            help="Ruta para guardar modelos."
        )
        parser.add_argument(
            "--buffer-size",
            type=int,
            default=2**16,
            help="Tamaño máximo del buffer de memoria de pasos temporales."
        )
        parser.add_argument(
            "--min-buffer-size",
            type=int,
            default=512,
            help="Ocupación mínima del buffer para realizar un entrenamiento."
        )
        parser.add_argument(
            "--n-batches",
            type=int,
            default=32,
            help="Número de mini batches por época de entrenamiento."
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="Tamaño del mini batch de entrenamiento."
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Número de épocas de cada ciclo de entrenamiento."
        )
        parser.add_argument(
            "--steps-to-train",
            type=int,
            default=2048,
            help="Número de steps necesarios que tienen que haberse ejecutado para realizar un entrenamiento desde el último."
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=5e-4,
            help="Learning rate"
        )
        parser.add_argument(
            "--max-grad-norm",
            type=float,
            default=0.5,
            help="Límite superior para realizar el clipping sobre la norma de los gradientes"
        )
        parser.add_argument(
            "--clipping-eps",
            type=float,
            default=0.2,
            help="Parámetro para estabelcer el límite inferior y superior para realizar el clipping sobre pérdida del actor en PPO"
        )
        parser.add_argument(
            "--entropy-coef",
            type=float,
            default=0.2,
            help="Coefiente para aplicar sobre el bonus de entropía en la pérdida del actor de PPO"
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.2,
            help="Factor de descuento de la funcion de utilidad"
        )
        parser.add_argument(
            "--gae-lambda",
            type=float,
            default=0.2,
            help="Factor de descuento del GAE"
        )
        parser.add_argument(
            "--monster-damage",
            type=float,
            default=-0.1,
            help="Daño a la integridad del agente que realiza el monstruo."
        )
        parser.add_argument(
            "--food-recovery",
            type=float,
            default=0.5,
            help="Recuperación de energía que provoca el consumo de comida."
        )
        parser.add_argument(
            "--medicine-recovery",
            type=float,
            default=1,
            help="Recuperación de integridad que provoca el consumo de medicina."
        )
        parser.add_argument(
            "--basal-cost",
            type=float,
            default=-0.001,
            help="Coste energético de quedarse quieto."
        )
        parser.add_argument(
            "--cost",
            type=float,
            default=-0.01,
            help="Coste energético de cualquier acción de movimiento."
        )
        parser.add_argument(
            "--action-recovery",
            type=float,
            default=0.05,
            help="Recuperación de integridad al realizar un acción de movimiento."
        )

        self._args = parser.parse_args()

        self._logger = ExperimentLogger()
        self._plotter = ExperimentPlotter(self._logger)

    def _run_training(self, agent, replay_buffer, training_config, training_count, device):
        print("###############################################")
        print(f"Comenzando el ciclo de entrenamiento {training_count}.")
        print("###############################################")

        print(f"Calculando los retornos y ventajas normalizadas para PPO.")

        self._logger.log_training_start()

        start = time.time()

        complete_buffer_data = replay_buffer.sample(len(replay_buffer))

        returns, advantages = PPOUtils.get_returns_and_advantages(
            complete_buffer_data["reward"],
            complete_buffer_data["value"],
            complete_buffer_data["done"],
            training_config.get_gamma(),
            training_config.get_gae_lambda()
        )

        complete_buffer_data["return"] = returns
        complete_buffer_data["advantage"] = advantages

        training_replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(training_config.get_replay_buffer_size()),
            sampler=SamplerWithoutReplacement(),
            batch_size=training_config.get_batch_size()
        )

        training_replay_buffer.extend(complete_buffer_data)

        interoception_prediction_loss = None

        for epoch in range(training_config.get_epochs()):
            print("-----------------------------------------------")
            print(f"Época {epoch + 1}:")
            print("-----------------------------------------------")

            for batch_num in range(training_config.get_n_batches()):
                batch = training_replay_buffer.sample()

                actor_loss, critic_loss, entropy = agent.train(batch)

                batch_str = f"Batch {batch_num + 1}/{training_config.get_n_batches()}. Actor loss: {actor_loss:.4f}, critic loss: {critic_loss:.4f}, entropy: {entropy:.4f}"

                print(f"DEBUG app.py -> intero arg: {self._args.intero}")
                if self._args.intero:
                    interoception_prediction_loss = agent.train_interoception_prediction(batch)
                    batch_str += f", interoception prediction loss: {interoception_prediction_loss:.4f}"

                print(batch_str)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.time()

        training_time = end - start
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        training_time = f"{hours} horas {minutes} minutos {seconds} segundos"

        print(f"Duración: {training_time}")

        self._logger.log_training_end(actor_loss.item(), critic_loss.item(), entropy.item(), interoception_prediction_loss)

    def _get_policy(self, policy, training_config):
        # TODO sacar las dimensiones programáticamente mediante los espacios de la observación del entorno
        policy_switch = {
            "basic": BasicPolicy(768, 5, training_config),
            "conv": ConvPolicy(5, training_config),
            "recurrent": RecurrentPolicy(768, 5, training_config),
            "convrec": ConvRecurrentPolicy(5, training_config),
            "random": RandomPolicy(5)
        }

        return policy_switch.get(policy, BasicPolicy(770, 5, training_config))

    def run(self):
        #TODO set the rest of the args.

        env = SurvivalEnv()

        training_config = TrainingConfig(
            self._args.buffer_size,
            self._args.min_buffer_size,
            self._args.n_batches,
            self._args.batch_size,
            self._args.epochs,
            self._args.steps_to_train,
            self._args.lr,
            self._args.max_grad_norm,
            self._args.clipping_eps,
            self._args.entropy_coef,
            self._args.gamma,
            self._args.gae_lambda
        )

        homeostasis_config = HomeostasisConfig(
            self._args.monster_damage,
            self._args.food_recovery,
            self._args.medicine_recovery,
            self._args.basal_cost,
            self._args.cost,
            self._args.action_recovery
        )

        policy = self._get_policy(self._args.policy, training_config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = Agent(
            [1.0, 1.0],
            policy,
            training_config,
            homeostasis_config,
            self._args.innate,
            self._args.intero,
            self._args.imagination,
            device
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(training_config.get_replay_buffer_size()),
            sampler=SamplerWithoutReplacement(),
        )

        reward_calculator = RewardCalculator()

        step_count = 0
        training_count = 0

        self._logger.log_experiment_start(self._args.episodes, self._args.policy, training_config, self._args.intero, self._args.imagination)

        for episode in range(0, self._args.episodes):
            done = False
            episode_duration = 0
            total_reward = 0
            food_count = 0
            medicine_count = 0
            damage_count = 0
            total_distance_to_monster = 0
            actions_count = [0,0,0,0,0]

            observation, info = env.reset()

            observation["interoception"] = agent.get_interoceptive_state()

            tensor_observation, tensor_reward = EnvTranformer.to_tensor(observation, device=device)

            while not done:
                predicted_interoception = agent.predict_interoception(
                    tensor_observation) if self._args.intero else None

                action, action_log_probabilities, entropy = agent.select_action(tensor_observation, info)

                value = agent.predict_value(tensor_observation)

                step_data = TensorDict(
                    {
                        "board": tensor_observation["board"].squeeze(),
                        "interoception": tensor_observation["interoception"].squeeze(),
                        "action": action.squeeze(),
                        "action_log_probabilities": action_log_probabilities.squeeze(),
                        "entropy": entropy.squeeze(),
                        "value": value.squeeze(),
                        "predicted_interoception": predicted_interoception
                    }
                )

                observation, _, _, _, info = env.step(action.item())

                previous_interoception = agent.get_interoceptive_state().copy()

                agent.update_interoception(info)

                observation["interoception"] = agent.get_interoceptive_state().copy()

                tensor_observation, _ = EnvTranformer.to_tensor(observation, 0, device)

                reward = reward_calculator.calculate_reward(
                    previous_interoception,
                    agent.get_interoceptive_state(),
                    predicted_interoception
                )

                _, tensor_reward = EnvTranformer.to_tensor(None, reward, device)

                done = agent.is_dead()

                step_data["reward"] = tensor_reward.squeeze()
                step_data["done"] = torch.tensor(done, device=device)
                step_data["next_interoception"] = tensor_observation["interoception"].squeeze()

                # No se añaden al replay buffer los steps en los que se han tomado acciones innatas para no desvirtuar el entrenamiento
                if not agent.is_last_action_innate() or not agent.is_last_action_conditioned():
                    print(f"DEBUG app.py -> last action innate: {agent.is_last_action_innate()}")
                    print(f"DEBUG app.py -> last action intero: {agent.is_last_action_conditioned()}")
                    print(f"DEBUG app.py -> previous intero: {previous_interoception}")
                    print(f"DEBUG app.py -> predicted intero: {predicted_interoception}")
                    print(f"DEBUG app.py -> current intero: {agent.get_interoceptive_state()}")
                    replay_buffer.add(step_data)

                if len(replay_buffer) >= training_config.get_min_buffer_size() and step_count % training_config.get_train_per_steps() == 0 and not done:
                    training_count += 1
                    self._run_training(agent, replay_buffer, training_config, training_count, device)

                step_count += 1
                episode_duration += 1
                total_reward += reward

                food_count += info["food"]
                medicine_count += info["medicine"]
                damage_count += info["damage"]
                total_distance_to_monster += info["distance_to_monster"]
                actions_count[action.item()] += 1

            agent.reset()

            mean_distance_to_monster = total_distance_to_monster / step_count
            mean_reward = total_reward / episode_duration

            print("################################")
            print(f"Resumen del episodio {episode + 1}.")
            print("################################")
            print(f"- Duración: {episode_duration} steps.")
            print(f"- Recompensa acumulada: {total_reward: .2f}.")
            print(f"- Recompensa media: {mean_reward:.2f}")
            print(f"- Comida ingerida: {food_count}")
            print(f"- Medicinas tomadas: {medicine_count}")
            print(f"- Mordiscos del monstruo recibidos: {damage_count}")
            print(f"- Distancia media al monstruo: {mean_distance_to_monster:.2f}")
            print(f"- Distribución de acciones: {actions_count}")

            self._logger.log_episode(episode_duration, total_reward, mean_reward, food_count, medicine_count, damage_count, mean_distance_to_monster, actions_count)

        self._logger.log_experiment_end()

        print("Guardando el resumen de métricas del experimento...")
        self._logger.save_to_json_file()

        print("Generando y guardando las gráficas de las métricas del experimento...")
        self._plotter.plot_all()

        env.close()

        print("################################")
        print("Experimento finalizado")
        print("################################")


