import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Type, Dict

from modules.downsample import DownSample
from modules.norm.layer_norm import LayerNorm
from modules.block.squeeze_excitation import SEBlock
from utils.load_module import load_module


class ConvDownSample(DownSample):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.scale_factor = int(self.scale_factor)

        self.down_layer = nn.Sequential(
            LayerNorm(self.in_channels),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
            ),
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=self.scale_factor,
                stride=self.scale_factor,
            ),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
            ) if self.in_channels != self.out_channels else
            nn.Identity(),
            LayerNorm(self.out_channels),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_layer(x) + self.shortcut(x)

