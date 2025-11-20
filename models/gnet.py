from typing import Tuple, Optional, Any, Union, Callable, List, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from dataclasses import dataclass, field
from omegaconf import DictConfig
import itertools, random

from models.model import Model
from modules.norm.layer_norm import LayerNorm2d


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class FusionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 ):
        super().__init__()

        fusion_dim = in_channels * 2

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_dim, in_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(in_channels),
            nn.GELU()
        )

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.fusion_conv(torch.cat([x0, x1], dim=1)).mean(dim=[-2, -1])


class GranularityNet(Model):
    @dataclass
    class Config(Model.Config):
        backbone: str = 'vgg'
        model_mapper: Dict = field(default_factory=lambda : {
            'vgg': 'vgg16_bn',
            'convnext': 'convnext_base',
            'convnextv2': 'convnextv2_base.fcmae_ft_in22k_in1k'
        })

    cfg: Config

    def __init__(self,
                 params: DictConfig,
                 ):

        super().__init__(params=params)

    def configure(self):
        super().configure()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if self.cfg.backbone not in self.cfg.model_mapper:
            raise ValueError(f"Unsupported model. please choose the following models: {list(self.cfg.model_mapper.keys())}")

        pretrained_weight_name = self.cfg.model_mapper[self.cfg.backbone]
        print(f"Build Model using Backbon: {pretrained_weight_name}.")
        self.backbone_img = timm.create_model(
            pretrained_weight_name, pretrained=True, features_only=True,
        )

        self.backbone_edge = timm.create_model(
            pretrained_weight_name, pretrained=True, features_only=True,
        )

        for param_img, param_edge in zip(self.backbone_img.parameters(), self.backbone_edge.parameters()):
            param_img.requires_grad = False
            param_edge.requires_grad = False

        feature_channels = self.backbone_img.feature_info.channels()

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(c) for c in feature_channels
        ])

        total_dims = sum(feature_channels)
        self.head = nn.Sequential(
            nn.Linear(total_dims, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, normalize_imagenet: bool = True) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        if normalize_imagenet:
            imgs = self.normalize_imagenet(imgs)
            edges = self.normalize_imagenet(edges)

        feats_imgs = self.backbone_img(imgs)
        feats_edges = self.backbone_edge(edges)

        fused_feats = []
        for fusion_block, feat_imgs, feat_edges in zip(self.fusion_blocks, feats_imgs, feats_edges):
            fused_feats.append(fusion_block(feat_imgs, feat_edges))

        global_feat = torch.cat(fused_feats, dim=1)

        return self.head(global_feat)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges, labels = batch

        granularity = self(imgs, edges)

        # loss, loss_log = self.loss_fn(granularity, labels, split='train' if self.training else 'valid')
        # self.log_dict(loss_log)

        loss = F.binary_cross_entropy(granularity, labels)
        split = 'train' if self.training else 'valid'
        self.log(f'{split}/loss', loss)

        return loss