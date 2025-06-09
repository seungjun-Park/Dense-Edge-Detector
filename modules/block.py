import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from utils.checkpoints import checkpoint


class Block(nn.Module, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 use_checkpoint: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels

        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)


    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass