import torch
import torch.nn as nn
from typing import List

from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights, VGG16_Weights
from torchvision.models.convnext import CNBlock
from thirdparty.convnext_v2 import ConvNeXtV2, Block

class VGG16(torch.nn.Module):
    def __init__(self,
                 ):
        super(VGG16, self).__init__()
        self.vgg_pretrained_features = vgg16(weights=VGG16_Weights).features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.vgg_pretrained_features(x)


class ConvNext(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        self.convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.convnext_features(x)

class ConvNextV2(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        net = ConvNeXtV2()
        net.load_state_dict(torch.load('models/backbone/convnextv2_tiny_22k_384_ema.pt', map_location=torch.device('cpu')), strict=False)

        net_stages = net.stages
        net_downsample_layers = net.downsample_layers

        self.features = nn.ModuleList()

        for i in range(4):
            self.features.append(net_downsample_layers[i])
            self.features.append(net_stages[i])

        self.features = nn.Sequential(*self.features)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.features(x)