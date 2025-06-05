import torch
import torch.nn as nn
from typing import Type, Dict

from utils import zero_module
from utils.load_module import load_module
from modules.block import Block
from utils.params import get_module_params


class MLP(Block):
    def __init__(self,
                 mlp_ratio: float | int = 2.0,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        embed_dim = int(self.in_channels * mlp_ratio)

        make_layer = load_module(self.layer)
        make_norm = load_module(self.norm)
        make_activation = load_module(self.activation)

        get_layer_params = get_module_params(make_layer)
        get_norm_params = get_module_params(make_norm)

        self.block = nn.Sequential(
            make_layer(**get_layer_params(self.in_channels, embed_dim, kernel_size=1)),
            make_norm(**get_norm_params(self.in_channels, num_groups=self.num_groups)),
            make_activation(),
            nn.Dropout(self.dropout),
            zero_module(
                make_layer(**get_layer_params(embed_dim, self.out_channels, kernel_size=1)),
            ),
            make_norm(**get_norm_params(embed_dim, num_groups=self.num_groups)),
            make_activation(),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path(self.block(x))


