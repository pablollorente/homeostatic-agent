import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=128):
        super(Generator, self).__init__()

        # Red compartida inicial que procesa el vector de ruido
        self.shared_network = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
        )

        # Rama para generar la imagen del tablero (16x16x3)
        self.image_branch = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 16 * 16 * 3),
            nn.Tanh()
        )

        # Rama para generar la interocepción (2 dimensiones)
        self.interoception_branch = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, noise):
        """
        Args:
            noise: tensor de forma [batch_size, noise_dim]

        Returns:
            imagined_board: tensor de forma [batch_size, 3, 16, 16]
            imagined_interoception: tensor de forma [batch_size, 2]
        """
        # Procesamiento compartido
        shared_features = self.shared_network(noise)

        # Generar imagen del tablero
        image_flat = self.image_branch(shared_features)
        # Convertir de Tanh [-1,1] a [0,1]
        image_flat = (image_flat + 1.0) / 2.0
        # Reshape a imagen 16x16x3
        imagined_board = image_flat.view(-1, 3, 16, 16)

        # Generar interocepción
        imagined_interoception = self.interoception_branch(shared_features)

        return imagined_board, imagined_interoception
