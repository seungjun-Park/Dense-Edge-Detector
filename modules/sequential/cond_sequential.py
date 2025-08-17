import torch
import torch.nn as nn


class ConditionalSequential(nn.Sequential):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        for block in self:
            x = block(x, cond)

        return x