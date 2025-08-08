import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.upsample import UpSample
from modules.norm.layer_norm import LayerNorm


class ConvUpSample(UpSample):
    def __init__(self,
                 mode: str = 'bilinear',
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        mode = mode.lower()
        assert mode in ['nearest', 'bilinear', 'bicubic', 'area', 'nearest-eaxct']
        self.mode = mode

        self.up_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels // self.scale_factor,
                kernel_size=3,
                padding=1,
            )
        )

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return self.up_layer(x)
