import torch
from abc import abstractmethod

from modules.block import Block


class DownSample(Block):
    def __init__(self,
                 scale_factor: float = 2.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.scale_factor = scale_factor

    @abstractmethod
    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        pass