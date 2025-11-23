import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers.grn import GlobalResponseNorm

from typing import Union, List, Tuple, Type, Dict, Callable

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block


class ResidualBlock(Block):
    def __init__(self,
                 in_channels: int,
                 embed_channels: int,
                 drop_path: float = 0.,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.embed = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_channels, in_channels * 8)
        )

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels , in_channels * 4, bias=False)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * in_channels, channels_last=True)
        self.pwconv2 = nn.Linear(in_channels * 4, in_channels, bias=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)

        if granularity is not None:
            granularity = self.embed(granularity).type(h.dtype)
            while len(granularity.shape) < len(x.shape):
                granularity = granularity[..., None]
            scale, shift = granularity.chunk(2, dim=1)
            h = h * (1 + scale) + shift

        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)

        h = h.permute(0, 3, 1, 2)

        return x + self.drop_path(h)
