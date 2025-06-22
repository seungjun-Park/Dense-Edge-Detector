import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from modules.upsample import UpSample
from modules.norm.layer_norm import LayerNorm


class ConvTransposeUpSample(UpSample):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.norm = LayerNorm(self.in_channels)

        self.conv_t = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.scale_factor,
            stride=self.scale_factor,
            padding=0,
            output_padding=0,
        )

    def _forward(self, x: torch.Tensor):
        return self.conv_t(self.norm(x))