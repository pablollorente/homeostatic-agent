import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Generador para la GAN de imaginación.

    Entrada: planning memory vector (128 dimensiones)
    Salida:
        - imagined_board: imagen RGB 16x16 (tensor [3, 16, 16])
        - imagined_interoception: vector 2D (tensor [2])
    """

    def __init__(self, planning_memory_dim=128):
        super(Generator, self).__init__()

        self.planning_memory_dim = planning_memory_dim

        # Red compartida inicial que procesa el planning memory
        self.shared_network = nn.Sequential(
            nn.Linear(planning_memory_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )

        # Rama para generar la imagen del tablero (16x16x3)
        self.image_branch = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 16 * 16 * 3),  # 768 neuronas para imagen 16x16x3
            nn.Tanh()  # Valores entre -1 y 1, luego se normalizarán a [0,1]
        )

        # Rama para generar la interocepción (2 dimensiones)
        self.interoception_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2),
            nn.Sigmoid()  # Valores entre 0 y 1 para la interocepción
        )

    def forward(self, planning_memory):
        """
        Args:
            planning_memory: tensor de forma [batch_size, 128]

        Returns:
            imagined_board: tensor de forma [batch_size, 3, 16, 16]
            imagined_interoception: tensor de forma [batch_size, 2]
        """
        # Procesamiento compartido
        shared_features = self.shared_network(planning_memory)

        # Generar imagen del tablero
        image_flat = self.image_branch(shared_features)
        # Convertir de Tanh [-1,1] a [0,1]
        image_flat = (image_flat + 1.0) / 2.0
        # Reshape a imagen 16x16x3
        imagined_board = image_flat.view(-1, 3, 16, 16)

        # Generar interocepción
        imagined_interoception = self.interoception_branch(shared_features)

        return imagined_board, imagined_interoception
