import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers.grn import GlobalResponseNorm

from typing import Union, List, Tuple, Type, Callable

from modules.block import Block


class ResidualBlock(Block):
    def __init__(self,
                 in_channels: int,
                 embed_channels: int,
                 drop_prob: float = 0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.embed = nn.Sequential(
            nn.GELU(approximate='tanh'),
            nn.Linear(embed_channels, in_channels * 2)
        )

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, in_channels * 4, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.grn = GlobalResponseNorm(4 * in_channels, channels_last=True)
        self.pwconv2 = nn.Linear(in_channels * 4, in_channels, bias=False)

        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        granularity = self.embed(granularity)
        granularity = granularity[:, None, None, :]

        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        scale, shift = granularity.chunk(2, dim=-1)
        h = h * (1 + scale) + shift

        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)

        h = h.permute(0, 3, 1, 2)

        return x + self.drop_path(h)
