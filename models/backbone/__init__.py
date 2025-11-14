import torch
import torch.nn as nn
from typing import List

from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights, VGG16_Weights
from torchvision.models.convnext import CNBlock
from thirdparty.convnext_v2 import ConvNeXtV2, Block

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
            self.slice1.append(nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            self.slice2.append(nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            self.slice3.append(nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            self.slice4.append(nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            self.slice5.append(nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))

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

        self.features_list = []

        self.slices = nn.ModuleList()
        for i in range(4):
            self.slices.append(nn.Sequential(*convnext_features[i * 2: i * 2 + 2]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if use_adaptor:
            for i, dim in enumerate([96, 192, 384, 768]):
                self.slices[i].append(CNBlock(dim, layer_scale=1e-6, stochastic_depth_prob=0.1))

        for i in range(4):
            self.slices[i][-1].block[4].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.features_list.append(outputs.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.features_list.clear()

        for s in self.slices:
            x = s(x)

        return self.features_list

class ConvNextV2(nn.Module):
    def __init__(self,
                 requires_grad: bool = False,
                 use_adaptor: bool = False
                 ):
        super().__init__()

        net = ConvNeXtV2()
        net.load_state_dict(torch.load('models/backbone/convnextv2_tiny_22k_384_ema.pt', map_location=torch.device('cpu')), strict=False)

        net_stages = net.stages
        net_downsample_layers = net.downsample_layers

        self.features_list = []

        self.slices = nn.ModuleList()
        for i in range(4):
            self.slices.append(nn.Sequential(net_downsample_layers[i], net_stages[i]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if use_adaptor:
            for i, dim in enumerate([96, 192, 384, 768]):
                self.slices[i].append(Block(dim, drop_path=0.1))

        for i in range(4):
            self.slices[i][-1].grn.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.features_list.append(outputs.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self.features_list.clear()

        for s in self.slices:
            x = s(x)

        return self.features_list