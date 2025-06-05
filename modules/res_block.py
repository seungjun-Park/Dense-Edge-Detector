import torch
import torch.nn as nn

from typing import Union, List, Tuple, Type, Dict, Callable

from transformers.activations import get_activation

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block
from utils.params import get_module_params


class ResidualBlock(Block):
    def __init__(self,
                 use_conv: bool = True,
                 *args,
                 **kwargs
                 ):
        self.use_conv = use_conv

        super().__init__(*args, **kwargs)

        make_layer = load_module(self.layer)
        make_norm = load_module(self.norm)
        make_activation = load_module(self.activation)

        get_layer_params = get_module_params(make_layer)
        get_norm_params = get_module_params(make_norm)

        self.block = nn.Sequential(
            self.make_layer(
                **get_layer_params(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)),
            make_norm(**get_norm_params(self.out_channels, num_groups=self.num_groups)),
            make_activation(),
            nn.Dropout(self.dropout),
            zero_module(
                make_layer(
                    *get_layer_params(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)),
            ),
            make_norm(**get_norm_params(self.out_channels, num_groups=self.num_groups)),
            make_activation(),
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif self.use_conv:
            self.shortcut = make_layer(**get_layer_params(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1))

        else:
            self.shortcut = make_layer(**get_layer_params(self.in_channels, self.out_channels, kernel_size=1))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.block(x)) + self.shortcut(x)


class ConvNextV2ResidualBlock(Block):
    def __init__(self,
                 embed_ratio: float | int = 4.0,
                 grn: str = "modules.norm.GlobalResponseNorm2d",
                 *args,
                 **kwargs
                 ):

        self.embed_ratio = embed_ratio
        self.make_grn = load_module(grn)

        super().__init__(*args, **kwargs)

    def _set_config(self) -> nn.Sequential:
        embed_dim = int(self.in_channels * self.embed_ratio)
        self.conv = self.make_layer(**self.make_layer_params(self.in_channels, self.in_channels, kernel_size=3, padding=1, groups=self.in_channels))
        self.block = nn.Sequential(
            self.make_norm(**self.make_norm_params(self.in_channels, eps=1e-6)),
            nn.Linear(self.in_channels, embed_dim),
            self.make_activation(),
            self.make_grn(embed_dim),
            nn.Linear(embed_dim, self.out_channels)
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.block(h)
        h = h.permute(0, 3, 1, 2)
        return self.drop_path(h) + x