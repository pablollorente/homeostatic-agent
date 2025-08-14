import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


class ExperimentPlotter:
    # TODO crear función para guardar las imágenes generadas por la GAN
    # TODO modificar todas las funciones de generación de gráficas para pintar la media móvil
    # TODO función para guardar la gráfica del número de acciones por episodio
    # TODO función para guardar las gráficas de las pérdidas y la entropía
    def __init__(self, logger, save_path = "experiments/plots"):
        self._experiment_log, self._episode_log, self._training_log = logger.get_log()
        self._save_path = save_path

        os.makedirs(save_path, exist_ok=True)

        self._figsize = (10, 6)
        self._dpi = 100

    def plot_duration_of_episodes(self):
        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["duration"], linewidth=2)
        plt.plot(self._get_moving_avg(self._episode_log["duration"]), color="b", alpha=0.5, linewidth=2)
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
        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["food_count"], color="g", linewidth=2, label='Comida consumida', alpha=0.8)
        plt.plot(self._episode_log["medicine_count"], color="b", linewidth=2, label='Medicina consumida', alpha=0.8)
        plt.plot(self._episode_log["damage_count"], color="r", linewidth=2, label='Mordiscos del monstruo', alpha=0.8)

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
        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["cumulative_reward"], linewidth=2, markersize=4)
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
        plt.figure(figsize=self._figsize)
        plt.plot(self._episode_log["avg_reward"], linewidth=2, markersize=4)
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
        training_rounds = range(len(self._training_log["actor_loss"]))

        plt.figure(figsize=self._figsize)
        plt.plot(training_rounds, self._training_log["actor_loss"], linewidth=2, markersize=4)
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
        training_rounds = range(len(self._training_log["critic_loss"]))

        plt.figure(figsize=self._figsize)
        plt.plot(training_rounds, self._training_log["critic_loss"], linewidth=2, markersize=4)
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

    def plot_entropy(self):
        training_rounds = range(len(self._training_log["entropy"]))

        plt.figure(figsize=self._figsize)
        plt.plot(training_rounds, self._training_log["entropy"], linewidth=2, markersize=4)
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

    def plot_all(self):
        self.plot_duration_of_episodes()
        self.plot_cumulative_reward()
        self.plot_avg_reward()
        self.plot_events()
        self.plot_avg_distance_to_monster()
        self.plot_actor_loss()
        self.plot_critic_loss()
        self.plot_entropy()
