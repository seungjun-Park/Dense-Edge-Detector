import torch
import torch.nn as nn
from timm.layers import DropPath

from typing import Union, List, Tuple, Type, Dict, Callable

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block
from modules.norm import GlobalResponseNorm


class ResidualBlock(Block):
    def __init__(self,
                 activation: str = 'torch.nn.GELU',
                 num_groups: int = 1,
                 use_conv: bool = True,
                 drop_path: float = 0.,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        make_activation = load_module(activation)

        self.block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, self.out_channels),
            make_activation(),
            nn.Dropout(self.dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            ),
            nn.GroupNorm(num_groups, self.out_channels),
            make_activation(),
        )

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            )

        else:
            self.shortcut = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.block(x)) + self.shortcut(x)


class ConvNextV2ResidualBlock(Block):
    def __init__(self,
                 activation: str = 'torch.nn.GELU',
                 embed_ratio: float | int = 4.0,
                 drop_path: float = 0.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        embed_dim = int(self.in_channels * embed_ratio)

        make_activation = load_module(activation)

        self.conv = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels
        )

        self.block = nn.Sequential(
            nn.LayerNorm(self.in_channels, eps=1e-6),
            nn.Linear(self.in_channels, embed_dim),
            make_activation(),
            GlobalResponseNorm(embed_dim),
            nn.Linear(embed_dim, self.out_channels)
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.block(h)
        h = h.permute(0, 3, 1, 2)
        return self.drop_path(h) + self.shortcut(x)