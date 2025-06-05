import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from timm.layers import DropPath

from utils.checkpoints import checkpoint


class Block(nn.Module, ABC):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 layer: str = 'torch.nn.Linear',
                 activation: str = None,
                 norm: str = None,
                 num_groups: int = None,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 use_checkpoint: bool = True,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels else in_channels
        self.layer = layer
        self.activation = activation
        self.norm = norm
        self.num_groups = num_groups
        self.dropout = dropout

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), flag=self.use_checkpoint)

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        pass