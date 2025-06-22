import torch
import torch.nn as nn

from utils.load_module import load_module
from modules.block import Block


class SEBlock(Block):
    def __init__(self,
                 embed_ratio: int = 1,
                 activation: str = 'torch.nn.ReLU',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        assert self.in_channels % embed_ratio == 0

        make_act = load_module(activation)

        embed_dim = int(self.in_channels // embed_ratio)

        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                self.in_channels,
                embed_dim,
                kernel_size=1,
            ),
            make_act(),
            nn.Conv2d(
                embed_dim,
                self.in_channels,
                kernel_size=1,
            ),
            nn.Sigmoid()
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)