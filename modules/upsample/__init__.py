import torch
from abc import abstractmethod

from modules.block import Block


class UpSample(Block):
    def __init__(self,
                 scale_factor: int | float = 2.0,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.scale_factor = scale_factor

    @abstractmethod
    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        pass