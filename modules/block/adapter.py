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

        # for i in range(num_blocks):
        #     self.layers.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
        #             nn.ReLU(),
        #             nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1),
        #         )
        #     )

        # for i in range(num_blocks):
        #     self.layers.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        #             nn.GroupNorm(1, in_channels),
        #             nn.GELU(),
        #             nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        #         )
        #     )

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
                    nn.GroupNorm(1, in_channels),
                    nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1,)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x

        return x