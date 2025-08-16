import torch
import torch.nn as nn
import math


class GaussianFourierEmbedding(nn.Module):
    def __init__(self,
                 dim: int,
                 sigma=4.0,
                 learnable=False,
                 ):
        super().__init__()
        # dim must be even; half for sin, half for cos
        assert dim % 2 == 0
        self.dim = dim
        D = dim // 2
        w = torch.randn(D) * sigma               # N(0, sigma^2)
        self.w = torch.nn.Parameter(w) if learnable else torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * x * self.w[None,:]  # [B, D]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, dim]