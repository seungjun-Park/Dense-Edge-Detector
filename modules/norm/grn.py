import torch
import torch.nn as nn


class GlobalResponseNorm(nn.Module):
    def __init__(self,
                 channels: int,
                 **ignored_keywords,
                 ):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch_size, ..., channels]
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-5)
        return self.gamma * (x * nx) + self.beta + x