import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Type

from utils.load_module import load_module
from utils.params import get_module_params
from modules.block import Block


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

        make_layer = load_module(self.layer)
        make_norm = load_module(self.norm)

        assert self.in_channels % self.out_channels == 0

        get_layer_params = get_module_params(make_layer)
        get_norm_params = get_module_params(make_norm)

        self.up_layer = nn.Sequential(
            make_norm(**get_norm_params(self.in_channels)),
            make_layer(**get_layer_params(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                groups=self.out_channels,
                **kwargs,
            ))
        )

    def _forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return self.up_layer(x)
