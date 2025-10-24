import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_blocks: int = 1,
                 ):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
                    nn.ReLU(),
                    nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x) + x