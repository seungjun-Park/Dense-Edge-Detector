import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.block import Block
from modules.norm import LayerNorm


class UpBlock(Block):
    def __init__(self,
                 scale_factor: int | float = 2.0,
                 mode: str = 'nearest',
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        mode = mode.lower()
        assert mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']
        self.mode = mode
        self.scale_factor = int(scale_factor)

        self.up_layer = nn.Sequential(
            nn.InstanceNorm2d(self.in_channels),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def _forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False, antialias=True)

        return self.up_layer(x)
