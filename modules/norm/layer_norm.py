import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self,
                 num_channels: int,
                 eps: float = 1e-6,
                 ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.normalize_shape = (num_channels, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.permute(0, 2, 3, 1)

        s = F.layer_norm(u, self.normalize_shape, self.weight, self.bias, self.eps)

        return s.permute(0, 3, 1, 2)