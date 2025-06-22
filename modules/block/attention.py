import math
from abc import abstractmethod

import torch
import torch.nn as nn
from flash_attn import flash_attn_qkvpacked_func

from modules.block import Block
from modules.block.mlp import MLP


class AttentionBlock(Block):
    def __init__(self,
                 num_heads: int = 1,
                 num_head_channels: int = None,
                 softmax_scale: float = None,
                 activation: str = 'torch.nn.GELU',
                 drop_path: float = 0.,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if num_head_channels is not None:
            assert self.in_channels % num_head_channels == 0
            self.num_heads = self.in_channels // num_head_channels
        else:
            self.num_heads = num_heads

        self.softmax_scale = (1 / math.sqrt(self.in_channels // self.num_heads)) if softmax_scale is None else softmax_scale

        self.qkv_layer = nn.Linear(self.in_channels, self.in_channels * 3)
        self.out_layer = nn.Linear(self.in_channels, self.in_channels)
        self.mlp_layer = MLP(
            activation=activation,
            mlp_ratio=4.0,
            drop_path=drop_path,
            **kwargs,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).permute(0, 2, 1)

        qkv = self.qkv_layer(x).reshape(b, -1, 3, self.num_heads, c // self.num_heads)
        out = self.out_layer(self.attn_func(qkv).reshape(b, -1, c))
        out = self.mlp_layer(out)
        out = out.permute(0, 2, 1).reshape(b, c, h, w)

        return out

    @abstractmethod
    def attn_func(self, qkv: torch.Tensor) -> torch.Tensor:
        pass


class MultiHeadAttentionBlock(AttentionBlock):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def attn_func(self, qkv: torch.Tensor) -> torch.Tensor:
        return


class FlashAttentionBlock(AttentionBlock):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def attn_func(self, qkv: torch.Tensor) -> torch.Tensor:
        return flash_attn_qkvpacked_func(qkv, 0., self.softmax_scale)


