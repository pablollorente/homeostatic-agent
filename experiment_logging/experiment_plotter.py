import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import os
from datetime import datetime


class ExperimentPlotter:
    # TODO pintar la media del mismo color que su gráfica y controlar cuando el array está vacío y no se puede calcular

    def __init__(self, logger, save_path = "experiments/plots"):
        self._experiment_log, self._episode_log, self._training_log = logger.get_log()
        self._save_path = os.path.join(save_path, self._experiment_log["id"])

        os.makedirs(self._save_path, exist_ok=True)

        self._figsize = (10, 6)
        self._dpi = 100

    def plot_duration_of_episodes(self):
        window_size = int(np.trunc(np.sqrt(self._experiment_log["episodes"])))

        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["duration"], color="b", alpha=0.4, linewidth=2)
        plt.plot(self._get_moving_avg(self._episode_log["duration"], window_size), color="b", alpha=0.8, linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Duración (steps)')
        plt.title('Duración en steps por episodio')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_duration_of_episodes_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_avg_distance_to_monster(self):
        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["avg_distance_to_monster"], linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Distancia (Manhattan)')
        plt.title('Distancia media al monstruo por episodio')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_avg_distance_to_monster_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_events(self):
        window_size = int(np.trunc(np.sqrt(self._experiment_log["episodes"])))

        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["food_count"], color="g", linewidth=2, label='Comida consumida', alpha=0.4)
        plt.plot(self._get_moving_avg(self._episode_log["food_count"], window_size), color="g", alpha=0.8, linewidth=2)
        plt.plot(self._episode_log["medicine_count"], color="b", linewidth=2, label='Medicina consumida', alpha=0.4)
        plt.plot(self._get_moving_avg(self._episode_log["medicine_count"], window_size), color="b", alpha=0.8, linewidth=2)
        plt.plot(self._episode_log["damage_count"], color="r", linewidth=2, label='Mordiscos del monstruo', alpha=0.4)
        plt.plot(self._get_moving_avg(self._episode_log["damage_count"], window_size), color="r", alpha=0.8, linewidth=2)

        plt.xlabel('Episodio')
        plt.ylabel('Cantidad')
        plt.title('Eventos por episodio')
        plt.legend()
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_events_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_cumulative_reward(self):
        window_size = int(np.trunc(np.sqrt(self._experiment_log["episodes"])))

        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["cumulative_reward"], color="b", alpha=0.4, linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._episode_log["cumulative_reward"], window_size), color="b", alpha=0.8, linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa acumulada')
        plt.title('Recompensa acumulada por episodio')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_cumulative_reward_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_avg_reward(self):
        window_size = int(np.trunc(np.sqrt(self._experiment_log["episodes"])))

        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["avg_reward"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._episode_log["avg_reward"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa media')
        plt.title('Recompensa media por episodio')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_avg_reward_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_actor_loss(self):
        training_rounds = len(self._training_log["actor_loss"])

        window_size = int(np.trunc(np.sqrt(training_rounds)))

        plt.figure(figsize=self._figsize)
        plt.plot(self._training_log["actor_loss"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["actor_loss"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Entrenamiento')
        plt.ylabel('Pérdida del actor')
        plt.title('Pérdida del actor por entrenamiento')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_actor_loss_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_critic_loss(self):
        training_rounds = len(self._training_log["critic_loss"])

        window_size = int(np.trunc(np.sqrt(training_rounds)))

        plt.figure(figsize=self._figsize)
        plt.plot(self._training_log["critic_loss"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["critic_loss"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Entrenamiento')
        plt.ylabel('Pérdida del crítico')
        plt.title('Pérdida del crítico por entrenamiento')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_critic_loss_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_interoception_loss(self):
        training_rounds = len(self._training_log["critic_loss"])

        window_size = int(np.trunc(np.sqrt(training_rounds)))

        plt.figure(figsize=self._figsize)
        plt.plot(self._training_log["interoception_prediction_loss"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["interoception_prediction_loss"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Entrenamiento')
        plt.ylabel('Pérdida')
        plt.title('Pérdida del predictor de interocepción')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_intero_loss_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_gan_loss(self):
        training_rounds = len(self._training_log["critic_loss"])

        window_size = int(np.trunc(np.sqrt(training_rounds)))

        plt.figure(figsize=self._figsize)
        plt.plot(self._training_log["generator_loss"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["generator_loss"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.plot(self._training_log["discriminator_loss"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["discriminator_loss"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Entrenamiento')
        plt.ylabel('Pérdida')
        plt.title('Pérdida del generador y del discriminador')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_gan_loss_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    def plot_entropy(self):
        training_rounds = len(self._training_log["entropy"])

        window_size = int(np.trunc(np.sqrt(training_rounds)))

        plt.figure(figsize=self._figsize)
        plt.plot(self._training_log["entropy"], alpha=0.4, color="b", linewidth=2, markersize=4)
        plt.plot(self._get_moving_avg(self._training_log["entropy"], window_size), alpha=0.8, color="b", linewidth=2)
        plt.xlabel('Entrenamiento')
        plt.ylabel('Entropía del actor')
        plt.title('Entropía del actor por entrenamiento')
        plt.grid(True, alpha=0.3)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"experiment_{self._experiment_log['id']}_entropy_{timestamp}.png"

        filepath = os.path.join(self._save_path, save_name)
        plt.savefig(filepath, dpi=self._dpi, bbox_inches="tight")
        plt.close()

        return filepath

    #TODO mejorar esta funcion para que coja la media de los anteriores puntos una vez que supera el índice en el que ya no se puede sumar la ventana porque se genera un outofbounds
    def _get_moving_avg(self, y, window = 10):
        avg_y = []
        for i in range(len(y) - window + 1):
            avg_y.append(np.mean(y[i:i + window]))

        return avg_y

    def save_gan_generated_image(self, rgb_array_image, step):
        save_name = f"training_round_{len(self._training_log['start_datetime']) + 1}_episode_{len(self._episode_log['duration']) + 1}_step_{step}.png"

        dirpath = os.path.join(self._save_path, "gan")

        os.makedirs(dirpath, exist_ok=True)

        filepath = os.path.join(dirpath, save_name)

        img.imsave(filepath, rgb_array_image)

    def plot_all(self):
        self.plot_duration_of_episodes()
        self.plot_cumulative_reward()
        self.plot_avg_reward()
        self.plot_events()
        self.plot_avg_distance_to_monster()
        self.plot_actor_loss()
        self.plot_critic_loss()
        self.plot_entropy()

        if self._experiment_log["interoception_prediction"]:
            self.plot_interoception_loss()

        if self._experiment_log["imagination"]:
            self.plot_gan_loss()
