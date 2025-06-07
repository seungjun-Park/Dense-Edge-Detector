import torch
import torch.nn as nn
from typing import Type, Dict

from timm.layers import DropPath

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block


class MLP(Block):
    def __init__(self,
                 activation: str = 'torch.nn.GELU',
                 mlp_ratio: float | int = 2.0,
                 drop_path: float = 0.,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        embed_dim = int(self.in_channels * mlp_ratio)

        make_activation = load_module(activation)

        self.block = nn.Sequential(
            nn.Linear(self.in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            make_activation(),
            zero_module(
                nn.Linear(embed_dim, self.out_channels),
            ),
            nn.LayerNorm(self.out_channels),
            make_activation(),
        )

        self.drop_path = DropPath(drop_path)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.block(x)) + x


