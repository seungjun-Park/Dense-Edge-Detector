import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 net_type: str = 'vgg',
                 num_blocks: int = 1,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.layers = nn.ModuleList()
        net_type = net_type.lower()
        assert net_type in ['vgg', 'convnext']

        self.last_activation = nn.ReLU() if net_type == 'vgg' else nn.GELU()
        mlp_ratio = 4 if net_type == 'vgg' else 2

        for i in range(num_blocks):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, int(in_channels * mlp_ratio), kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(int(in_channels * mlp_ratio), in_channels, kernel_size=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x) + x

        return self.last_activation(x)
