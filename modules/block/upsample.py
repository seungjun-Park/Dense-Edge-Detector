import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.block import Block

from modules.norm.layer_norm import LayerNorm2d


class Upsample(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        out_channels = out_channels if out_channels else in_channels

        self.norm = LayerNorm2d(in_channels)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return self.conv(x)
