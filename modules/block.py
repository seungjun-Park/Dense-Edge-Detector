import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from torch.utils.checkpoint import checkpoint


class Block(nn.Module, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 use_checkpoint: bool = False,
                 device_type: str = 'cuda',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.use_checkpoint = use_checkpoint
        self.device_type = device_type.lower()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
                return checkpoint(self._forward, x)

        return self._forward(x)


    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass