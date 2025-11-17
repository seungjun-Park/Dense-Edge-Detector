from typing import Tuple, Optional, Any, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from dataclasses import dataclass, field
from omegaconf import DictConfig
import itertools, random

from models.model import Model
from models.backbone import VGG16, ConvNext, ConvNextV2


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)


class LPIEPG(Model):
    @dataclass
    class Config(Model.Config):
        backbone: str = 'vgg'
        stage_2_start_epoch: int = 100
        pairs: List = field(default_factory=lambda: list(itertools.combinations([0, 1, 2], 2)))

    cfg: Config

    def __init__(self,
                 params: DictConfig,
                 ):

        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.cfg.backbone = self.cfg.backbone.lower()

        assert self.cfg.backbone in ['vgg', 'convnext', 'convnext_v2']

        self.scaling_layer = ScalingLayer()
        self.lins = nn.ModuleList()

        if self.cfg.backbone == 'vgg':
            self.backbone = VGG16()
            for dim in [64, 128, 256, 512, 512]:
                self.lins.append(
                    NetLinLayer(dim, use_dropout=True)
                )

        elif self.cfg.backbone == 'convnext':
            self.backbone = ConvNext()
            for dim in [96, 192, 384, 768]:
                self.lins.append(
                    NetLinLayer(dim * 4, use_dropout=True)
                )

        else:
            self.backbone = ConvNextV2()
            for dim in [96, 192, 384, 768]:
                self.lins.append(
                    NetLinLayer(dim * 4, use_dropout=True)
                )

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.backbone(imgs, True)
        feats_edges = self.backbone(edges)

        val = 0

        for i, (feat_imgs, feat_edges) in enumerate(zip(feats_imgs, feats_edges)):
            diff = (normalize_tensor(feat_imgs) - normalize_tensor(feat_edges)) ** 2
            val += spatial_average(self.lins[i](diff), keepdim=True)

        return val

    def step(self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor]], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges = batch
        if self.current_epoch < self.cfg.stage_2_start_epoch:
            return self._stage_1(imgs, edges[0])

        if self.current_epoch == self.cfg.stage_2_start_epoch:
            for p in self.backbone.adaptors.parameters():
                p.requires_grad = False
        idx = random.randint(0, 2)
        pair = self.cfg.pairs[idx]

        if idx == 1:
            margin = 1.0
        else:
            margin = 0.5

        edges_pos, edges_neg = edges[pair[0]], edges[pair[1]]

        return self._stage_2(imgs, edges_pos, edges_neg, margin)

    def _stage_1(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.backbone(imgs, True)
        feats_edges = self.backbone(edges)

        loss = 0
        loss_log = {}
        split = 'train' if self.training else 'valid'

        for i, (feat_imgs, feat_edges) in enumerate(zip(feats_imgs, feats_edges)):
            loss_lv = F.mse_loss(normalize_tensor(feat_imgs), normalize_tensor(feat_edges))
            loss_log.update({f'{split}/loss_lv{i}': loss_lv})
            loss += loss_lv

        loss_log.update({f'{split}/loss': loss})
        self.log_dict(loss_log)

        return loss

    def _stage_2(self, imgs: torch.Tensor, edges_pos: torch.Tensor, edges_neg: torch.Tensor, margin: float) -> torch.Tensor:
        d_pos = self(imgs, edges_pos)
        d_neg = self(imgs, edges_neg)

        loss = self.loss_fn(d_pos, d_neg, margin).mean()

        split = 'train' if self.training else 'valid'
        self.log(f'{split}/loss', loss)

        return loss

class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.TensorType:
        return self.model(x)