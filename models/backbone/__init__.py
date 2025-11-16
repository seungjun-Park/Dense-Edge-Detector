import torch
import torch.nn as nn
from typing import List

from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights, VGG16_Weights
from torchvision.models.convnext import CNBlock
from thirdparty.convnext_v2 import ConvNeXtV2, Block

class VGG16(torch.nn.Module):
    def __init__(self,
                 requires_grad=False,
                 ):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights).features

        self.slices = nn.ModuleList()
        self.adaptors = nn.ModuleList()

        self.slices.append(nn.Sequential(*vgg_pretrained_features[0: 4]))
        self.slices.append(nn.Sequential(*vgg_pretrained_features[4: 9]))
        self.slices.append(nn.Sequential(*vgg_pretrained_features[9: 16]))
        self.slices.append(nn.Sequential(*vgg_pretrained_features[16: 23]))
        self.slices.append(nn.Sequential(*vgg_pretrained_features[23: 30]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        for dim in [64, 128, 256, 512, 512]:
            self.adaptors.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor, use_adaptor: bool = False) -> List[torch.Tensor]:
        features = []
        for i in range(5):
            x = self.slices[i](x)
            if use_adaptor:
                x = self.adaptors[i](x)
            features.append(x)

        return features


class ConvNext(nn.Module):
    def __init__(self,
                 requires_grad: bool = False,
                 ):
        super().__init__()

        convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

        self.features_list = []
        self.adaptor_features_list = []

        self.slices = nn.ModuleList()
        self.adaptors = nn.ModuleList()
        for i in range(4):
            self.slices.append(nn.Sequential(*convnext_features[i * 2], *convnext_features[i * 2 + 1]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        for i, dim in enumerate([96, 192, 384, 768]):
            self.adaptors.append(CNBlock(dim, layer_scale=1e-6, stochastic_depth_prob=0.1))

        for i in range(4):
            self.slices[i][-1].block[4].register_forward_hook(self.hook_fn)
            self.adaptors[i].block[4].register_forward_hook(self.adaptor_hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.features_list.append(outputs.permute(0, 3, 1, 2))

    def adaptor_hook_fn(self, module, inputs, outputs):
        self.adaptor_features_list.append(outputs.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor, use_adaptor: bool = False) -> List[torch.Tensor]:
        self.features_list.clear()
        self.adaptor_features_list.clear()

        for i in range(4):
            x = self.slices[i](x)
            if use_adaptor:
                x = self.adaptors[i](x)

        return self.adaptor_features_list.copy() if use_adaptor else self.features_list.copy()

class ConvNextV2(nn.Module):
    def __init__(self,
                 requires_grad: bool = False,
                 ):
        super().__init__()

        net = ConvNeXtV2()
        net.load_state_dict(torch.load('models/backbone/convnextv2_tiny_22k_384_ema.pt', map_location=torch.device('cpu')), strict=False)
        net_stages = net.stages
        net_downsample_layers = net.downsample_layers

        self.features_list = []
        self.adaptor_features_list = []

        self.slices = nn.ModuleList()
        self.adaptors = nn.ModuleList()
        for i in range(4):
            self.slices.append(nn.Sequential(*net_downsample_layers[i], *net_stages[i]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        for i, dim in enumerate([96, 192, 384, 768]):
            self.adaptors.append(Block(dim, drop_path=0.1))

        for i in range(4):
            self.slices[i][-1].grn.register_forward_hook(self.hook_fn)
            self.adaptors[i].grn.register_forward_hook(self.adaptor_hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.features_list.append(outputs.permute(0, 3, 1, 2))

    def adaptor_hook_fn(self, module, inputs, outputs):
        self.adaptor_features_list.append(outputs.permute(0, 3, 1, 2))

    def forward(self, x: torch.Tensor, use_adaptor: bool = False) -> List[torch.Tensor]:
        self.features_list.clear()
        self.adaptor_features_list.clear()

        for i in range(4):
            x = self.slices[i](x)
            if use_adaptor:
                x = self.adaptors[i](x)

        return self.adaptor_features_list.copy() if use_adaptor else self.features_list.copy()