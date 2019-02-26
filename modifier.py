import torch
from torch import nn


class Modifier(nn.Module):

    def __init__(self, z_size=64):
        super().__init__()
        self.z_size = z_size

        self.model = nn.Sequential(
            nn.Linear(z_size+102, 8*z_size),
            nn.SELU(),
            nn.Linear(8*z_size, 8*z_size),
            nn.SELU(),
            nn.Linear(8 * z_size, 8 * z_size),
            nn.SELU(),
            nn.Linear(8 * z_size, 8 * z_size),
            nn.SELU(),
            nn.Linear(8*z_size, z_size),
            nn.Tanh()
        )

    def forward(self, x, age, gender):
        inp = torch.cat([x, age, gender], dim=1)
        return self.model(inp)


