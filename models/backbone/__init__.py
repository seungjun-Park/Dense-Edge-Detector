import torch
import torch.nn as nn
from typing import List

from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights, VGG16_Weights
from thirdparty.convnext_v2 import ConvNeXtV2


class VGG16(torch.nn.Module):
    def __init__(self,
                 requires_grad=False,
                 ):
        super(VGG16, self).__init__()
        self.vgg_pretrained_features = vgg16(weights=VGG16_Weights).features.eval()
        for param in self.vgg_pretrained_features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vgg_pretrained_features(x)


class ConvNext(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        self.convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features.eval()
        for param in self.convnext_features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnext_features(x)

        return x


class ConvNextV2(nn.Module):
    def __init__(self,
                 ):
        super().__init__()

        net = ConvNeXtV2().eval()
        net.load_state_dict(torch.load('models/backbone/convnextv2_tiny_22k_384_ema.pt', map_location=torch.device('cpu')), strict=False)
        net_stages = net.stages
        net_downsample_layers = net.downsample_layers

        self.convnext_v2_features = nn.ModuleList()

        for i in range(4):
            self.convnext_v2_features.append(nn.Sequential(*net_downsample_layers[i], *net_stages[i]))

        self.convnext_v2_features = nn.Sequential(*self.convnext_v2_features)

        for param in self.convnext_v2_features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext_v2_features(x)