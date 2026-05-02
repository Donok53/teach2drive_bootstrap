from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.05


class RoutePolicyMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        layers = []
        dim = config.input_dim
        for _ in range(config.depth):
            layers.extend([
                nn.Linear(dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            ])
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class SensorFusionPolicy(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        output_dim: int,
        image_channels: int = 3,
        lidar_channels: int = 3,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.image_encoder = ConvEncoder(image_channels, embed_dim)
        self.lidar_encoder = ConvEncoder(lidar_channels, embed_dim)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, scalar, image, lidar):
        fused = torch.cat([
            self.scalar_encoder(scalar),
            self.image_encoder(image),
            self.lidar_encoder(lidar),
        ], dim=1)
        return self.head(fused)

