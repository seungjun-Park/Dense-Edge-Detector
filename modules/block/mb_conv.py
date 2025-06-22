import torch
import torch.nn as nn
from timm.layers import DropPath

from utils.load_module import load_module
from modules.block import Block
from modules.norm.layer_norm import LayerNorm
from modules.block.squeeze_excitation import SEBlock


class FusedMBConvBlock(Block):
    def __init__(self,
                 embed_ratio: int | float = 4.0,
                 activation: str = 'torch.nn.ReLU',
                 drop_path: float = 0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        embed_dim = int(self.in_channels * embed_ratio)

        make_act = load_module(activation)

        self.block = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm(embed_dim),
            make_act(),
            SEBlock(
                in_channels=embed_dim,
                embed_ratio=2,
            ),
            nn.Conv2d(
                embed_dim,
                self.out_channels,
                kernel_size=1,
            ),
            LayerNorm(self.out_channels),
            make_act(),
        )

        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()

        if self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
            )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.block(x)) + self.shortcut(x)