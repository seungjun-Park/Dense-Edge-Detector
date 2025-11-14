import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from omegaconf import DictConfig

from models.model import Model
from models.backbone import VGG16, ConvNext


class LMIES(Model):
    @dataclass
    class Config(Model.Config):
        in_channels: int = 3
        backbone: str = 'vgg'

    cfg: Config

    def __init__(self,
                 params: DictConfig,
                 ):
        super().__init__(params=params)

    def configure(self):
        super().configure()

        if self.cfg.backbone == 'vgg':
            self.backbone_imgs = VGG16(use_adaptor=True)
            self.backbone_edges = VGG16().eval()
        else:
            self.backbone_imgs = ConvNext(use_adaptor=True)
            self.backbone_edges = ConvNext().eval()

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        feats_imgs = self.backbone_imgs(imgs)
        feats_edges = self.backbone_edges(edges)

        return feats_imgs, feats_edges


    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.ShortTensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges_0, edges_1, edges_2 = batch
        feats_imgs, feats_edges = self(imgs, edges_0)

        split = 'train' if self.training else 'valid'
        loss_dict = {}
        loss = 0
        for i, (feat_imgs, feat_edges) in enumerate(zip(feats_imgs, feats_edges)):
            loss_lv = F.mse_loss(feat_imgs, feat_edges)
            loss_dict.update({f'{split}/loss_lv{i}': loss_lv.clone().detach().mean()})
            loss += loss_lv

        loss_dict.update({f'{split}/loss': loss.clone().detach().mean()})
        self.log_dict(loss_dict)

        return loss


