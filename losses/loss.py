import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict


class Loss(nn.Module, ABC):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pass