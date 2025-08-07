class Interoception():
    def __init__(self, initial_interoceptive_state):
        self._initial_interoceptive_state = initial_interoceptive_state
        self._interoceptive_state = initial_interoceptive_state.copy()

    def get_interoceptive_state(self):
        return self._interoceptive_state

    def add_energy(self, a):
        self._interoceptive_state[0] += a
        self._interoceptive_state[0] = min(1, max(0, self._interoceptive_state[0]))

    def add_integrity(self, a):
        self._interoceptive_state[1] += a
        self._interoceptive_state[1] = min(1, max(0, self._interoceptive_state[1]))

    def set_integrity_to_max(self):
        self._interoceptive_state[1] = 1

    def is_dead(self):
        return self._interoceptive_state[0] <= 0 or self._interoceptive_state[1] <= 0