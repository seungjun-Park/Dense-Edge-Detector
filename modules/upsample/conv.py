import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.upsample import UpSample


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

        self.up_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return self.up_layer(x)
