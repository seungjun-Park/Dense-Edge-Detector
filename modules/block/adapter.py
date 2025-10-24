import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)