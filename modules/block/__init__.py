import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.checkpoints import checkpoint
from modules.norm.layer_norm import LayerNorm


class Block(nn.Module, ABC):
    def __init__(self,
                 use_checkpoint: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = use_checkpoint

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, *args) -> torch.Tensor:
        return checkpoint(self._forward, (*args,), self.parameters(), flag=self.use_checkpoint)


    @abstractmethod
    def _forward(self, *args) -> torch.Tensor:
        pass