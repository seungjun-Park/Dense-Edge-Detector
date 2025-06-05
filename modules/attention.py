import math
from abc import abstractmethod

import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func

from modules.block import Block
from utils.load_module import load_module
from utils.params import get_module_params


class _AttentionBlock(Block):
    def __init__(self,
                 qkv_layer: str = 'torch.nn.Linear',
                 out_layer: str = 'torch.nn.Linear',
                 num_heads: int = 1,
                 num_head_channels: int = None,
                 softmax_scale: float = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if num_head_channels is not None:
            assert self.in_channels % num_head_channels == 0
            self.num_heads = self.in_channels // num_head_channels
        else:
            self.num_heads = num_heads

        self.softmax_scale = (1 / math.sqrt(self.in_channels // self.num_heads)) if softmax_scale is None else softmax_scale

        make_qkv_layer = load_module(qkv_layer)
        make_out_layer = load_module(out_layer)

        get_qkv_layer_params = get_module_params(make_qkv_layer)
        get_out_layer_params = get_module_params(make_out_layer)


        self.qkv_layer = make_qkv_layer(**get_qkv_layer_params(self.in_channels, self.in_channels * 3, **kwargs))
        self.out_layer = make_out_layer(**get_out_layer_params(self.in_channels, self.in_channels, **kwargs))

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MultiHeadAttentionBlock(_AttentionBlock):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return


class FlashAttentionBlock(_AttentionBlock):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(flash_attn_qkvpacked_func(self.qkv_layer(x), self.dropout, self.softmax_scale))
