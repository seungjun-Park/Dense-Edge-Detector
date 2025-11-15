from typing import Tuple, Optional, Any, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.optimizer import LightningOptimizer
from dataclasses import dataclass, field
from omegaconf import DictConfig

from models.model import Model
from models.backbone import VGG16, ConvNext, ConvNextV2
from thirdparty.convnext_v2 import GRN


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)

def upsample(in_tens: torch.Tensor, out_HW: Tuple[int] = (64, 64)) -> torch.Tensor:
    in_H, in_w = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

class Permute(nn.Module):
    def __init__(self,
                 dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)


class LPIEPG(Model):
    @dataclass
    class Config(Model.Config):
        backbone: str = 'vgg'

    cfg: Config

    def __init__(self,
                 params: DictConfig,
                 ):

        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.cfg.backbone = self.cfg.backbone.lower()

        assert self.cfg.backbone in ['vgg', 'convnext', 'convnext_v2']

        if self.cfg.backbone == 'vgg':
            self.img_enc = VGG16().eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False
            self.edge_enc = VGG16().features

            self.img_mlp = nn.Sequential(
                nn.Conv2d(512, 2048, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 512, kernel_size=1)
            )

            self.edge_mlp = nn.Sequential(
                nn.Conv2d(512, 2048, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, 512, kernel_size=1)
            )

        elif self.cfg.backbone == 'convnext':
            self.img_enc = ConvNext().eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False
            self.edge_enc = ConvNext()

            self.img_mlp = nn.Sequential(
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(768, eps=1e-6),
                nn.Linear(768, 768 * 4),
                nn.GELU(),
                nn.Linear(768 * 4, 512),
                Permute((0, 3, 1, 2))
            )

            self.edge_mlp = nn.Sequential(
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(768, eps=1e-6),
                nn.Linear(768, 768 * 4),
                nn.GELU(),
                nn.Linear(768 * 4, 512),
                Permute((0, 3, 1, 2))
            )

        else:
            self.img_enc = ConvNextV2().eval()
            for p in self.img_enc.parameters():
                p.requires_grad = False
            self.edge_enc = ConvNextV2()

            self.img_mlp = nn.Sequential(
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(768, eps=1e-6),
                nn.Linear(768, 768 * 4),
                nn.GELU(),
                GRN(768 * 4),
                nn.Linear(768 * 4, 512),
                Permute((0, 3, 1, 2))
            )

            self.edge_mlp = nn.Sequential(
                Permute((0, 2, 3, 1)),
                nn.LayerNorm(768, eps=1e-6),
                nn.Linear(768, 768 * 4),
                nn.GELU(),
                GRN(768 * 4),
                nn.Linear(768 * 4, 512),
                Permute((0, 3, 1, 2))
            )

        self.scaling_layer = ScalingLayer()

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        imgs = self.scaling_layer(imgs)

        z_imgs = self.img_mlp(self.img_enc(imgs))
        z_edges = self.edge_mlp(self.edge_enc(edges))

        diff = (z_imgs - z_edges) ** 2

        return diff

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges_0, edges_1, margin = batch

        d0 = self(imgs, edges_0)
        d1 = self(imgs, edges_1)

        loss = self.loss_fn(d0, d1, margin).mean()

        split = 'train' if self.training else 'valid'
        self.log(f'{split}/loss', loss, prog_bar=True)

        return loss


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.shift) / self.scale
