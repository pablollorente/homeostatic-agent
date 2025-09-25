import numpy as np


class RewardCalculator:
    """
    Calcula recompensas homeostáticas basadas en la distancia al punto de equilibrio
    y la diferencia entre lo reducción de distancia al equilibrio real y la imaginada ponderada.

    La fórmula de la recompensa es: r = real - coef * imaginada

    Donde:
    - real: es la reducción de la distancia al equilibrio homeostático [1,1] en el estado real
    - imaginada: es la reducción de la distancia "imaginada" por el agente
    """

    def __init__(self, pred_intero_coef = 0.5, dist_to_equilibrium_coef = 5e-3):
        """
        Inicializa el calculador de recompensas.
        """
        # Punto de equilibrio homeostático [energía, integridad]
        self._homeostatic_equilibrium = np.array([1.0, 1.0])
        self._pred_intero_coef = pred_intero_coef
        self._dist_to_equilibrium_coef = dist_to_equilibrium_coef

    def _get_distance_to_equilibrium(self, interoceptive_state):
        """
        Calcula la distancia euclidiana entre el estado interoceptivo y el punto de equilibrio homeostático

        Args:
            interoceptive_state: Vector interoceptivo

        Returns:
            distance: Distancia al punto
        """
        return np.linalg.norm(interoceptive_state - self._homeostatic_equilibrium)

    def _get_distance_reduction(self, previous_interoception, current_interoception):
        """
        Calcula la reducción de la distancia al equilibrio entre dos estados.
        Un valor positivo indica que el agente se ha acercado al equilibrio.

        Args:
            previous_interoception: Estado interoceptivo anterior
            current_interoception: Estado interoceptivo actual

        Returns:
            reduction: Reducción de la distancia
        """
        prev_distance = self._get_distance_to_equilibrium(previous_interoception)
        current_distance = self._get_distance_to_equilibrium(current_interoception)

        # Reducción de distancia (positiva si nos acercamos al equilibrio)
        reduction = prev_distance - current_distance

        return reduction

    def calculate_reward(self, previous_interoception, current_interoception, predicted_interoception = None):
        """
        Calcula la recompensa homeostática según la fórmula: r = (real - coef * imaginada) * 10.
        real - coef * imaginada se llama delta y, cuando no hay predicción de la interocepion delta = real.
        A delta se le suma la división entre un coeficiente y la distancia al equilibrio, para dar mayores recompensas
        cuando el agente está cerca del equilibrio, aunque pierda energía o integridad.

        Args:
            previous_interoception: Estado interoceptivo real anterior
            current_interoception: Estado interoceptivo real actual
            predicted_interoception: Estado interoceptivo predicho

        Returns:
            reward: Recompensa homeostática
        """
        # Calcular reducción de distancia real
        real_reduction = self._get_distance_reduction(previous_interoception, current_interoception)

        if predicted_interoception is not None:
            # Calcular reducción de distancia imaginada
            predicted_reduction = self._get_distance_reduction(previous_interoception, predicted_interoception)

            # Calcular recompensa según la fórmula
            delta = real_reduction - self._pred_intero_coef * predicted_reduction
        else:
            delta = real_reduction

        sign = np.sign(delta)

        # Acotamos el rango de delta al máximo de distancia entre el origen del vector [0, 0] y el final [1, 1], que es sqrt(2)
        # ya que, cuando se utiliza la predicción de la interocepción no hay límite al valor predicho.
        delta_clipped = np.clip(np.absolute(delta), 0, np.sqrt(2))

        distance_to_equilibrium = self._get_distance_to_equilibrium(current_interoception)

        delta_scaled = sign * delta_clipped + self._dist_to_equilibrium_coef / (distance_to_equilibrium + 1e-3)

        reward = delta_scaled * 10

        return reward.item()
