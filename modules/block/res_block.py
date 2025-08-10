import torch
import torch.nn as nn
from timm.layers import DropPath

from typing import Union, List, Tuple, Type, Dict, Callable

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block
from modules.norm.layer_norm import LayerNorm


class ResidualBlock(Block):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 activation: str = 'torch.nn.GELU',
                 use_conv: bool = True,
                 drop_path: float = 0.,
                 num_groups: int = 1,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        out_channels = out_channels if out_channels else in_channels

        make_activation = load_module(activation)

        self.in_block = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            make_activation(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.out_block = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            make_activation(),
            zero_module(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            ),
        )

        self.embed_granularity = nn.Sequential(
            nn.Linear(
                1,
                out_channels * 2,
            ),
            make_activation(),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()

        elif use_conv:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )

        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        h = self.in_block(x)
        granularity = self.embed_granularity(granularity).type(h.dtype)
        while len(granularity.shape) < len(h.shape):
            granularity = granularity[..., None]
        out_norm, out_rest = self.out_block[0], self.out_block[1:]
        scale, shift = granularity.chunk(2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        return self.drop_path(h) + self.shortcut(x)
