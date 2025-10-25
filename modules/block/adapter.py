import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 block_type: str = 'mlp',
                 num_blocks: int = 1,
                 use_residual: bool = False,
                 ):
        super().__init__()

        self.layers = nn.ModuleList()
        block_type = block_type.lower()
        assert block_type in ['mlp', 'res_block', 'cn_block']

        self.use_residual = use_residual

        for i in range(num_blocks):
            if block_type == 'mlp':
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1),
                    )
                )

            elif block_type == 'res_block':
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(1, in_channels),
                        nn.GELU(),
                        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
                    )
                )

            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels),
                        nn.GroupNorm(1, in_channels),
                        nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
                        nn.GELU(),
                        nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1)
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(x)
            if self.use_residual:
                x = h + x
            else:
                x = h
        return x