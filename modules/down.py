import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Type, Dict

from modules.block import Block


class DownBlock(Block):
    def __init__(self,
                 scale_factor: int | float = 2.0,
                 num_groups: int = 1,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        scale_factor = int(scale_factor)

        assert self.out_channels % self.in_channels == 0

        self.down_layer = nn.Sequential(
            nn.GroupNorm(num_groups, self.in_channels),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=scale_factor,
                stride=scale_factor,
                groups=self.in_channels,
            )
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_layer(x)

