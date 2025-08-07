import gymnasium as gym
import numpy as np
import pygame

from gymnasium import spaces
from interoception import Interoception
from monster import Monster


class SurvivalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512
        self.cell_size = self.window_size // 16

        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=255, shape=(16, 16, 3), dtype=np.uint8),
            'interoception': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(5)

        self._board = None
        self._agent_position = None
        self._monster = None
        self._agent = Interoception([1,1])

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + 50))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size + 50))
        canvas.fill((0, 0, 0))

        # Draw grid
        for i in range(16):
            for j in range(16):
                pygame.draw.rect(
                    canvas,
                    (255, 255, 255),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    0
                )
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size),
                    1
                )

        # Draw food (green circles)
        for i in range(16):
            for j in range(16):
                if self._board[i, j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (0, 255, 0),
                        (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2),
                        self.cell_size // 8
                    )

        # Draw medicines (blue circles)
        for i in range(16):
            for j in range(16):
                if self._board[i, j] == 2:
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 255),
                        (j * self.cell_size + self.cell_size // 2, i * self.cell_size + self.cell_size // 2),
                        self.cell_size // 8
                    )

        # Draw agent (black square)
        agent_pos = self._agent_position
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            (int(agent_pos[1]) * self.cell_size + self.cell_size // 4,
             int(agent_pos[0]) * self.cell_size + self.cell_size // 4,
             self.cell_size // 2,
             self.cell_size // 2)
        )

        # Draw monster (red square)
        monster_position = self._monster.get_position()
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            (int(monster_position[1]) * self.cell_size + self.cell_size // 4,
             int(monster_position[0]) * self.cell_size + self.cell_size // 4,
             self.cell_size // 2,
             self.cell_size // 2)
        )

        # Draw stats
        font = pygame.font.Font(None, 36)
        stats_text = f"Energy: {self._agent.get_interoceptive_state()[0]:.2f} Integrity: {self._agent.get_interoceptive_state()[1]:.2f}"
        text_surface = font.render(stats_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(self.window_size // 2, self.window_size + 25))
        canvas.blit(text_surface, text_rect)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _render_as_rgb_array(self):
        # Create 16x16x3 array with white background
        frame = np.ones((16, 16, 3), dtype=np.uint8) * 255

        # Set food positions (green)
        food_positions = np.where(self._board == 1)
        frame[food_positions[0], food_positions[1]] = [0, 255, 0]

        # Set medicine positions (blue)
        medicine_positions = np.where(self._board == 2)
        frame[medicine_positions[0], medicine_positions[1]] = [0, 0, 255]

        # Set agent position (black)
        frame[tuple(self._agent_position)] = [0, 0, 0]

        # Set monster position (red)
        frame[tuple(self._monster.get_position())] = [255, 0, 0]

        return frame

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_as_rgb_array()
        elif self.render_mode == "human":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._board = np.zeros((16, 16), dtype=np.int8)

        # Coloca 12 comidas en los espacios vacíos del tablero
        empty_positions = list(zip(*np.where(self._board == 0)))
        food_positions = self.np_random.choice(len(empty_positions), size=48, replace=False)
        for pos in food_positions:
            self._board[empty_positions[pos]] = 1

        # Coloca 3 medicinas en los espacios vacíos del tablero que han quedado tras colocar las comidas
        empty_positions = list(zip(*np.where(self._board == 0)))
        medicine_positions = self.np_random.choice(len(empty_positions), size=12, replace=False)
        for pos in medicine_positions:
            self._board[empty_positions[pos]] = 2

        # Posiciona al agente en el tablero
        empty_positions = list(zip(*np.where(self._board == 0)))
        initial_agent_position = empty_positions[self.np_random.choice(len(empty_positions))]
        self._agent_position = initial_agent_position

        # Place monster
        empty_positions = list(zip(*np.where(self._board == 0)))
        initial_monster_position = empty_positions[self.np_random.choice(len(empty_positions))]
        self._monster = Monster(np.array(initial_monster_position, dtype=np.int8), 0.6)

        observation = {
            'board': self._render_as_rgb_array(),
            'interoception': self._agent.get_interoceptive_state()
        }

        info = {
            'damage': False,
            'food': False,
            'medicine': False,
            'distance_to_monster': self._get_agent_distance_to_monster()
        }

        return observation, info

    def step(self, action):
        # Calcular la nueva posición del agente en función de la acción relizada
        self._agent_position = self._get_new_agent_position(action)
        new_position = tuple(self._agent_position)

        # Mover al monstruo
        self._monster.move(self._agent_position)

        # Construir la observación
        observation = {
            'board': self._render_as_rgb_array(),
            'interoception': self._agent.get_interoceptive_state()
        }

        info = {
            'damage': np.array_equal(self._agent_position, self._monster.get_position()),
            'food': self._check_for_item_and_regenerate(new_position, 1),
            'medicine': self._check_for_item_and_regenerate(new_position, 2),
            'distance_to_monster': self._get_agent_distance_to_monster()
        }

        return observation, None, False, False, info

    def _get_new_agent_position(self, action):
        new_position = list(self._agent_position).copy()

        if action == 1:  # up
            new_position[0] = max(0, self._agent_position[0] - 1)
        elif action == 2:  # down
            new_position[0] = min(15, self._agent_position[0] + 1)
        elif action == 3:  # left
            new_position[1] = max(0, self._agent_position[1] - 1)
        elif action == 4:  # right
            new_position[1] = min(15, self._agent_position[1] + 1)

        return new_position

    def _check_for_item_and_regenerate(self, position, type=1):
        if self._board[position] == type:
            self._board[position] = 0

            # Hacer reaparecer la comida o la medicina
            if self.np_random.random() < 0.9:
                empty_positions = list(zip(*np.where(self._board == 0)))
                if empty_positions:
                    new_item_position = empty_positions[self.np_random.choice(len(empty_positions))]
                    self._board[new_item_position] = type

            return True

        return False
    
    def _get_agent_distance_to_monster(self):
        """
        Calcula la distancia Manhattan entre el agente y el monstruo.
        
        Returns:
            int: Distancia Manhattan
        """
        agent_pos = self._agent_position
        monster_pos = self._monster.get_position()
        return abs(agent_pos[0] - monster_pos[0]) + abs(agent_pos[1] - monster_pos[1])
