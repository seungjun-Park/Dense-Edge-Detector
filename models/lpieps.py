from typing import Tuple, Optional, Any, Union, Callable, List, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from dataclasses import dataclass, field
from omegaconf import DictConfig
import itertools, random

from models import DefaultModel
from modules.norm.layer_norm import LayerNorm2d


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class Adaptor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction_ratio: float = 2
                 ):
        super().__init__()

        hidden_dim = int(in_channels // reduction_ratio)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None ,None])
        self.register_buffer('scale', torch.Tensor([-.458, -.448, -.450])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LPIEPS(DefaultModel):
    def __init__(self,
                 params: DictConfig,
                 ):

        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.backbone = timm.create_model('vgg16_bn', pretrained=True, features_only=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_channels = self.backbone.feature_info.channels()

        self.scaling_layer = ScalingLayer()
        self.lins = nn.ModuleList()
        self.adaptors = nn.ModuleList()

        for c in feature_channels:
            self.lins.append(
                NetLinLayer(c)
            )

            self.adaptors.append(
                Adaptor(c)
            )

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, return_feats: bool = False) -> torch.Tensor:
        b = imgs.shape[0]
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.backbone(imgs)
        feats_edges = self.backbone(edges)

        val = 0
        adaptors_feats = []
        for adaptor, lin, feat_imgs, feat_edges in zip(self.adaptors, self.lins, feats_imgs, feats_edges):
            feat_imgs = adaptor(feat_imgs)
            adaptors_feats.append(feat_imgs)
            diff = (normalize_tensor(feat_imgs) - normalize_tensor(feat_edges)) ** 2
            res = spatial_average(lin(diff))
            val += res

        if return_feats:
            return  val.reshape(b), adaptors_feats, feats_edges

        return val.reshape(b)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges_0, edges_1, edges_2 = batch

        d_high, adaptors_feats, feats_edges = self(imgs, edges_0, True)
        d_mid = self(imgs, edges_1)
        d_poor = self(imgs, edges_2)

        loss, loss_log = self.loss_fn(d_high, d_mid, d_poor, adaptors_feats, feats_edges, split='train' if self.training else 'valid')
        self.log_dict(loss_log)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'lins' in name:
                    param.clamp_(min=1e-6)