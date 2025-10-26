import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.convnext_v2 import LayerNorm


class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 net_type: str = 'vgg',
                 num_blocks: int = 1,
                 dropout: float = 0.1,
                 ):
        super().__init__()

        self.layers = nn.ModuleList()
        net_type = net_type.lower()
        assert net_type in ['vgg', 'convnext']

        self.last_activation = nn.ReLU() if net_type == 'vgg' else nn.GELU()

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    LayerNorm(in_channels, data_format='channels_first'),
                    nn.Conv2d(in_channels, int(in_channels * 4), kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(int(in_channels * 4), in_channels, kernel_size=1),
                    nn.Dropout(dropout)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x

        return self.last_activation(x)