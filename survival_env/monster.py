import random


class Monster:
    """
    Monstruo que persigue al agente en el entorno para intentar dañarle.
    """

    def __init__(self, initial_position, staying_still_prob):
        """
        Inicializa al monstruo en la posición del entorno indicada en position

        Args:
            position: Posición inicial del monstruo en el entorno.
        """
        self._position = initial_position
        self._staying_still_prob = staying_still_prob

    def _get_action(self, agent_position):
        if random.uniform(0,1) < self._staying_still_prob:
            return 0

        # Calcula la dirección en la que se debe mover hacia el agente
        diff = agent_position - self._position

        # Movimiento vertical
        if abs(diff[0]) > abs(diff[1]):
            return 1 if diff[0] < 0 else 2
        # Movimiento horizontal
        else:
            return 3 if diff[1] < 0 else 4

    def move(self, agent_position):
        action = self._get_action(agent_position)

        if action == 1:  # arriba
            self._position[0] = max(0, self._position[0] - 1)
        elif action == 2:  # abajo
            self._position[0] = min(15, self._position[0] + 1)
        elif action == 3:  # izquierda
            self._position[1] = max(0, self._position[1] - 1)
        elif action == 4:  # derecha
            self._position[1] = min(15, self._position[1] + 1)

    def get_position(self):
        return self._position
