import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Discriminador para la GAN de imaginación.

    Entrada:
        - board: imagen RGB 16x16 (tensor [3, 16, 16])
        - interoception: vector 2D (tensor [2])

    Salida: probabilidad de que la entrada sea real (tensor [1])
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # Encoder convolucional para la imagen
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 16x16x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 8x8x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 4x4x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # 1x1x256
            nn.LeakyReLU(0.2),
            nn.Flatten()  # 256 features
        )

        # Red que combina imagen e interocepción
        self.classifier = nn.Sequential(
            # 256 (imagen) + 2 (interocepción) = 258
            nn.Linear(256 + 2, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilidad entre 0 y 1
        )

    def forward(self, board, interoception):
        """
        Args:
            board: tensor de forma [batch_size, 3, 16, 16]
            interoception: tensor de forma [batch_size, 2]

        Returns:
            probability: tensor de forma [batch_size, 1] - probabilidad de ser real
        """
        # Procesar imagen
        image_features = self.image_encoder(board)

        # Concatenar imagen e interocepción
        combined_features = torch.cat([image_features, interoception], dim=1)

        # Clasificar como real/fake
        probability = self.classifier(combined_features)

        return probability
