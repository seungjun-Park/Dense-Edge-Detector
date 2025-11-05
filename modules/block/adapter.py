import torch
import torch.nn as nn

from thirdparty.convnext_v2 import LayerNorm, GRN

class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 net_type: str = 'vgg',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        net_type = net_type.lower()
        assert net_type in ['vgg', 'convnext']

        self.last_activation = nn.ReLU() if net_type == 'vgg' else nn.GELU()
        mlp_ratio = 4 if net_type == 'vgg' else 2

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=3, groups=in_channels),
            LayerNorm(in_channels, eps=1e-6),
            nn.Conv2d(in_channels, int(in_channels * mlp_ratio), kernel_size=1),
            nn.GELU(),
            GRN(int(in_channels * mlp_ratio)),
            nn.Conv2d(int(in_channels * mlp_ratio), in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.last_activation(self.layer(x))
