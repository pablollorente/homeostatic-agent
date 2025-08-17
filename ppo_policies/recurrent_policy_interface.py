from abc import ABC, abstractmethod


class RecurrentPolicyInterface(ABC):
    @abstractmethod
    def get_last_actor_h(self):
        pass

    @abstractmethod
    def get_last_critic_h(self):
        pass
