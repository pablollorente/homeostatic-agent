import torch
import numpy as np


class EnvTranformer:

    @staticmethod
    def to_tensor(observation, reward = 0, device = "cpu"):
        board = observation["board"]

        # Normalizar valores de los píxeles
        board = board.astype(np.float32) / 255

        # Convertir a tensor y cambiar al formato de pytorch: [batch_size, canales, altura, ancho]
        board = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(device)

        # Convertir la lista de interocepción en un tensor
        interoception = torch.FloatTensor(observation["interoception"]).unsqueeze(0).to(device)

        observation = {
            "board": board,
            "interoception": interoception
        }

        reward = torch.FloatTensor(reward).unsqueeze(0).to(device)

        return observation, reward
