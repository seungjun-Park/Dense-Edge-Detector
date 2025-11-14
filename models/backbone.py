import torch
import torch.nn as nn
from typing import List

from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights, VGG16_Weights
from modules.block.adaptor import Adaptor

class VGG16(torch.nn.Module):
    def __init__(self,
                 requires_grad=False,
                 use_adaptor: bool = False,
                 ):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights).features
        self.slice1 = nn.Sequential(*vgg_pretrained_features[0: 4])
        self.slice2 = nn.Sequential(*vgg_pretrained_features[4: 9])
        self.slice3 = nn.Sequential(*vgg_pretrained_features[9: 16])
        self.slice4 = nn.Sequential(*vgg_pretrained_features[16: 23])
        self.slice5 = nn.Sequential(*vgg_pretrained_features[23: 30])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if use_adaptor:
            self.slice1.append(Adaptor(64, net_type='vgg'))
            self.slice2.append(Adaptor(128, net_type='vgg'))
            self.slice3.append(Adaptor(256, net_type='vgg'))
            self.slice4.append(Adaptor(512, net_type='vgg'))
            self.slice5.append(Adaptor(512, net_type='vgg'))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]


class ConvNext(nn.Module):
    def __init__(self,
                 requires_grad: bool = False,
                 use_adaptor: bool = False
                 ):
        super().__init__()

        convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

        self.slice1 = nn.Sequential(*convnext_features[0: 2])
        self.slice2 = nn.Sequential(*convnext_features[2: 4])
        self.slice3 = nn.Sequential(*convnext_features[4: 6])
        self.slice4 = nn.Sequential(*convnext_features[6: 8])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if use_adaptor:
            self.slice1.append(Adaptor(96, net_type='convnext'))
            self.slice2.append(Adaptor(192, net_type='convnext'))
            self.slice3.append(Adaptor(384, net_type='convnext'))
            self.slice4.append(Adaptor(768, net_type='convnext'))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)

        return [h1, h2, h3, h4]