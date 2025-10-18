from typing import Tuple, Optional, Any, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from torchvision.models import vgg16, convnext_tiny, ConvNeXt_Tiny_Weights

from models.model import Model
from utils.init_weight import init_weight
from utils.load_module import load_module


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIEPSV2(Model):
    def __init__(self,
                 net_type: str = 'vgg',
                 use_dropout: bool = True,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.net_type = net_type.lower()
        assert self.net_type in ['vgg', 'convnext']

        if self.net_type == 'vgg':
            self.net_imgs = VGG16().eval()
            self.net_edges = VGG16(requires_grad=True)
            self.chns = [64, 128, 256, 512, 512]

        elif self.net_type == 'convnext':
            self.net_imgs = ConvNext().eval()
            self.net_edges = ConvNext(requires_grad=True)
            self.chns = [96, 192, 384, 768]

        self.scaling_layer = ScalingLayer()
        self.lins = nn.ModuleList()
        self.moderators_imgs = nn.ModuleList()
        self.moderators_edges = nn.ModuleList()

        for i in range(len(self.chns)):
            self.lins.append(
                NetLinLayer(self.chns[i], use_dropout=use_dropout)
            )
            self.moderators_imgs.append(
                Moderator(self.chns[i], net_type)
            )
            self.moderators_edges.append(
                Moderator(self.chns[i], net_type)
            )

        self.save_hyperparameters(ignore=['loss_config'])

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
        moderators = []

        for i, (feat_imgs, feat_edges) in enumerate(zip(feats_imgs, feats_edges)):
            feat_imgs = self.moderators[i](feat_imgs)
            moderators.append(feat_imgs)

        return feats_imgs, moderators, feats_edges

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        val = 0

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.net_imgs(imgs)
        feats_edges = self.net_edges(edges)

        for i in range(len(feats_imgs)):
            feat_imgs = self.moderators_imgs[i](feats_imgs[i])
            feat_edges = self.moderators_edges[i](feats_edges[i])
            diff = (normalize_tensor(feat_imgs) - normalize_tensor(feat_edges)) ** 2
            res = spatial_average(self.lins[i](diff), keepdim=True)

            val += res

        return val


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

        loss = self.loss(d0, d1, labels) + F.mse_loss(d_high, 0)

        split = 'train' if self.training else 'valid'
        self.log(f'{split}/loss', loss, prog_bar=True)

        self.log(f'{split}/d_high', d_high.mean(), prog_bar=True)
        self.log(f'{split}/d_low', d_low.mean(), prog_bar=True)

        return loss

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Union[Optimizer, LightningOptimizer],
            optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        for param in self.lins.parameters():
            param.data.clamp_(min=0)


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


class Moderator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 net_type: str = 'vgg',
                 ):
        super().__init__()

        if net_type == 'vgg':
            make_activation = torch.nn.ReLU(inplace=True)
        else:
            make_activation = torch.nn.GELU()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            make_activation,
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
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
    def __init__(self, requires_grad: bool = False):
        super().__init__()

        convnext_features = convnext_tiny(ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

        self.slices = nn.ModuleList()

        for i in range(4):
            self.slices.append(convnext_features[i * 2: i * 2 + 2])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor):
        feats = []

        for s in self.slices:
            x = s(x)
            feats.append(x)

        return feats
