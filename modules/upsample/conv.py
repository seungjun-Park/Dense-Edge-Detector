import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.upsample import UpSample
from modules.norm.layer_norm import LayerNorm2d


class ConvUpSample(UpSample):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 mode: str = 'bilinear',
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        mode = mode.lower()
        assert mode in ['nearest', 'bilinear', 'bicubic', 'area', 'nearest-eaxct']
        self.mode = mode

        out_channels = out_channels if out_channels else in_channels

        self.norm = LayerNorm2d(in_channels)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        x = self.norm(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return self.conv(x)
