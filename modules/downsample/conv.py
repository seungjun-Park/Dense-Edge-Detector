import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Type, Dict

from modules.downsample import DownSample
from modules.norm.layer_norm import LayerNorm


class ConvDownSample(DownSample):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.scale_factor = int(self.scale_factor)

        self.down_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                int(self.in_channels * self.scale_factor),
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
            ),
        )

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        return self.down_layer(x)

