from typing import Tuple, Optional, Any, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.convnext import CNBlock

from models.model import Model
from utils.init_weight import init_weight


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIEPSV3(Model):
    def __init__(self,
                 net_type: str = 'vgg',
                 net_requires_grad: bool = False,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.net_type = net_type.lower()
        assert self.net_type in ['vgg', 'convnext']

        if self.net_type == 'vgg':
            self.net = VGG16(requires_grad=net_requires_grad)
            self.chns = [64, 128, 256, 512, 512]

        elif self.net_type == 'convnext':
            self.net = ConvNext(requires_grad=net_requires_grad)
            self.chns = [96, 192, 384, 768]

        self.scaling_layer = ScalingLayer()

        if not net_requires_grad:
           self.net = self.net.eval()

        self.save_hyperparameters(ignore=['loss_config', 'net_requires_grad'])

    def _get_features(self, imgs: torch.Tensor, edges: torch.Tensor, normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalize:
            imgs = 2 * imgs - 1
            edges = 2 * edges - 1

        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.net(imgs)
        feats_edges = self.net(edges)

        return feats_imgs, feats_edges

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feat_imgs = self.net(imgs)
        feat_edges = self.net(edges)

        diff = F.pairwise_distance(feat_imgs, feat_edges, p=2, keepdim=True)

        return diff


    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges_0, edges_1, labels = batch

        d0 = self(imgs, edges_0)
        d1 = self(imgs, edges_1)

        d_high = torch.zeros_like(d0).to(d0.device)
        d_high[labels == 0.0] = d0[labels == 0.0]
        d_high[labels == 1.0] = d1[labels == 1.0]

        d_low = torch.zeros_like(d0).to(d0.device)
        d_low[labels != 1.0] = d1[labels != 1.0]
        d_low[labels != 0.0] = d0[labels != 0.0]

        loss = self.loss(d0, d1, labels)

        split = 'train' if self.training else 'valid'
        self.log(f'{split}/loss', loss, prog_bar=True)

        self.log(f'{split}/d_high', d_high.mean(), prog_bar=True)
        self.log(f'{split}/d_low', d_low.mean(), prog_bar=True)

        return loss


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self,
                 chn_in: int,
                 chn_out: int = 1,
                 use_dropout: bool = False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, dropout: float = 0.):
        super(VGG16, self).__init__()
        self.vgg_pretrained_features = vgg16(pretrained=pretrained).features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg_pretrained_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)

        return x


class ConvNext(nn.Module):
    def __init__(self, requires_grad: bool = False, dropout: float = 0.,):
        super().__init__()

        self.convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(384, 384),
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convnext_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp(x)

        return x
