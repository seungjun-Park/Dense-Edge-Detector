import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.upsample import UpSample
from modules.norm.layer_norm import LayerNorm
from utils.load_module import load_module


class PixelShuffleUpSample(UpSample):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.sub_pix_conv = nn.Sequential(
            LayerNorm(self.in_channels),
            nn.Conv2d(
                self.in_channels,
                self.in_channels * (self.scale_factor ** 2),
                kernel_size=3,
                padding=1,
            ),
        )

        self.pixel_shuffle = nn.PixelShuffle(int(self.scale_factor))

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
        )

    def _forward(self, x: torch.Tensor):
        return self.conv(self.pixel_shuffle(self.sub_pix_conv(x)))