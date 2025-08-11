import torch
import torch.nn as nn

from modules.block import Block

class FiLM(Block):
    def __init__(self,
                 in_channels: int,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.gamma_beta = nn.Conv2d(in_channels, in_channels * 2, 1)

    def _forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gamma_beta(cond).chunk(2, dim=1)

        return x * gamma + beta