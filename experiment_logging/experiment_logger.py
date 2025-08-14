import json
import numpy as np
import os
import time
import uuid


class ExperimentLogger:
    def __init__(self):
        self._experiment_log = {
            "id": str(uuid.uuid4()),
            "start_datetime": None,
            "end_datetime": None,
            "agent_type": None,
            "episodes": None,
            "training_configuration": None
        }

        self._episode_log = {
            "duration": np.array([]),
            "cumulative_reward": np.array([]),
            "avg_reward": np.array([]),
            "food_count": np.array([]),
            "medicine_count": np.array([]),
            "damage_count": np.array([]),
            "avg_distance_to_monster": np.array([]),
        }

        self._training_log = {
            "start_datetime": np.array([]),
            "end_datetime": np.array([]),
            "actor_loss": np.array([]),
            "critic_loss": np.array([]),
            "entropy": np.array([]),
        }

    def log_experiment_start(self, episodes, agent_type, training_conf):
        self._experiment_log["start_datetime"] = time.localtime()
        self._experiment_log["episodes"] = episodes
        self._experiment_log["agent_type"] = agent_type
        self._experiment_log["training_configuration"] = training_conf.__dict__

    def log_experiment_end(self):
        self._experiment_log["end_datetime"] = time.strftime("%Y-%M-%d %H:%M:%S", time.localtime())

    def log_episode(
        self,
        duration,
        cumulative_reward,
        avg_reward,
        food_count,
        medicine_count,
        damage_count,
        avg_distance_to_monster
    ):
        self._episode_log["duration"] = np.append(self._episode_log["duration"], duration)
        self._episode_log["cumulative_reward"] = np.append(self._episode_log["cumulative_reward"], cumulative_reward)
        self._episode_log["avg_reward"] = np.append(self._episode_log["avg_reward"], avg_reward)
        self._episode_log["food_count"] = np.append(self._episode_log["food_count"], food_count)
        self._episode_log["medicine_count"] = np.append(self._episode_log["medicine_count"], medicine_count)
        self._episode_log["damage_count"] = np.append(self._episode_log["damage_count"], damage_count)
        self._episode_log["avg_distance_to_monster"] = np.append(self._episode_log["avg_distance_to_monster"], avg_distance_to_monster)

    def log_training_start(self):
        self._training_log["start_datetime"] = np.append(self._training_log["start_datetime"], time.strftime("%Y-%M-%d %H:%M:%S", time.localtime()))

    def log_training_end(self, actor_loss, critic_loss, entropy):
        self._training_log["actor_loss"] = np.append(self._training_log["actor_loss"], actor_loss)
        self._training_log["critic_loss"] = np.append(self._training_log["critic_loss"], critic_loss)
        self._training_log["entropy"] = np.append(self._training_log["entropy"], entropy)
        self._training_log["end_datetime"] = np.append(self._training_log["end_datetime"], time.strftime("%Y-%M-%d %H:%M:%S", time.localtime()))

    def save_to_json_file(self, path = "experiments/logs"):
        log = {
            "experiment": self._experiment_log,
            "episodes": [],
            "training_rounds": []
        }

        for index in range(self._experiment_log["episodes"]):

            episode = {
                "number": index,
                "duration": self._episode_log["duration"][index],
                "cumulative_reward": self._episode_log["cumulative_reward"][index],
                "avg_reward": self._episode_log["avg_reward"][index],
                "food_count": self._episode_log["food_count"][index],
                "medicine_count": self._episode_log["medicine_count"][index],
                "damage_count": self._episode_log["damage_count"][index],
                "avg_distance_to_monster": self._episode_log["avg_distance_to_monster"][index],
            }

            log["episodes"].append(episode)

        for index in range(len(self._training_log["start_datetime"])):
            training_round = {
                "number": index,
                "start_datetime": self._training_log["start_datetime"][index],
                "end_datetime": self._training_log["end_datetime"][index],
                "actor_loss": self._training_log["actor_loss"][index],
                "critic_loss": self._training_log["critic_loss"][index],
                "entropy": self._training_log["entropy"][index],
            }

            log["training_rounds"].append(training_round)

        filename = (f"{time.strftime('%Y%M%d_%H%M%S', self._experiment_log['start_datetime'])}"
                    f"_experiment_{self._experiment_log['agent_type']}"
                    f"_{str(self._experiment_log['episodes'])}"
                    f"_{self._experiment_log['id']}.json"
        )

        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, filename)

        self._experiment_log["start_datetime"] = time.strftime("%Y-%M-%d %H:%M:%S", self._experiment_log["start_datetime"])

        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(log, file, indent=4, ensure_ascii=False)

    def get_log(self):
        return self._experiment_log, self._episode_log, self._training_log

