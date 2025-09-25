from abc import ABC, abstractmethod


class ImaginationPolicy(ABC):
    @abstractmethod
    def get_last_h(self):
        pass
