import torch
import torch.nn as nn

from torchvision.models.convnext import CNBlock


class Adaptor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 net_type: str = 'vgg',
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        net_type = net_type.lower()
        assert net_type in ['vgg', 'convnext']

        if net_type == 'vgg':
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        else:
            self.layer = nn.Sequential(
                CNBlock(in_channels, layer_scale=1e-6, stochastic_depth_prob=0.1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
