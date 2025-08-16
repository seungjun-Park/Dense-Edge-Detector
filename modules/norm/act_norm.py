import torch
import torch.nn as nn


# ---------------------------
# ActNorm (data-dependent init)
# ---------------------------
class ActNorm2d(nn.Module):
    def __init__(self,
                 dim: int,
                 eps=1e-6,
                 ):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps
        self.initialized = False

    @torch.no_grad()
    def _init(self, x):
        m = x.mean((0, 2, 3), keepdim=True)
        s = x.std((0, 2, 3), keepdim=True).clamp_min(self.eps)
        self.bias.copy_(-m)
        self.logs.copy_(-s.log())
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init(x)
        return (x + self.bias) * torch.exp(self.logs)
