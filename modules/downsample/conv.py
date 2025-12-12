import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Type, Dict

from modules.norm.layer_norm import LayerNorm2d
from modules.downsample import DownSample


class ConvDownSample(DownSample):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        out_channels = out_channels if out_channels else in_channels

        self.down_layer = nn.Sequential(
            LayerNorm2d(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
        )

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        return self.down_layer(x)

