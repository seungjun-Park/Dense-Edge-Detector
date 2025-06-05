import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Type, Dict

from modules.block import Block
from utils.params import get_module_params
from utils.load_module import load_module


class DownBlock(Block):
    def __init__(self,
                 scale_factor: int | float = 2.0,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.scale_factor = int(scale_factor)

        assert self.out_channels % self.in_channels == 0

        make_layer = load_module(self.layer)
        make_norm = load_module(self.norm)

        get_norm_params = get_module_params(make_norm)
        get_layer_params = get_module_params(make_layer)

        self.down_layer = nn.Sequential(
            make_norm(**get_norm_params(self.in_channels)),
            make_layer(
                **get_layer_params(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                    groups=self.in_channels,
                    **kwargs,
                ))
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_layer(x)

