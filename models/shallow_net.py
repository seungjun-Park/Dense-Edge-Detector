from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Tuple, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig

from models import DefaultModel
from modules.norm.layer_norm import LayerNorm2d
from utils import granularity_embedding
from modules.block.res_block import CNBlockV2


class ShallowNet(DefaultModel):
    @dataclass
    class Config(DefaultModel.Config):
        in_channels: int = 3
        embed_dim: int = 32
        out_channels: int = None
        num_blocks: int = 2
        drop_prob: float = 0.2
        use_checkpoint: bool = True

    cfg: Config

    def __init__(self,
                 params: DictConfig
                 ):
        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.loss_fn = self.loss_fn.eval()

        out_channels = self.cfg.out_channels if self.cfg.out_channels is not None else self.cfg.in_channels

        granularity_embed_dim = self.cfg.embed_dim * 4
        self.granularity_embed = nn.Sequential(
            nn.Linear(self.cfg.embed_dim, granularity_embed_dim),
            nn.GELU(),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.embed_block = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.in_channels, kernel_size=7, padding=3, bias=False),
            nn.Conv2d(self.cfg.in_channels, self.cfg.embed_dim, kernel_size=1, bias=False),
            LayerNorm2d(self.cfg.embed_dim),
        )

        self.layers = nn.ModuleList()
        for i in range(self.cfg.num_blocks):
            self.layers.append(
                CNBlockV2(
                    self.cfg.embed_dim,
                    granularity_embed_dim,
                    drop_path=self.cfg.drop_prob,
                    use_checkpoint=self.cfg.use_checkpoint
                )
            )

    def forward(self, x: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        granularity = granularity_embedding(granularity, dim=self.cfg.embed_dim)
        emb = self.granularity_embed(granularity)
        x = self.embed_block(x)
        for layer in self.layers:
            x = layer(x, emb)

        x = F.sigmoid(x)

        return torch.min(x, dim=1, keepdim=True)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges, labels = batch

        preds = self(imgs, labels)

        loss, loss_log = self.loss_fn(imgs, preds, edges, labels, split='train' if self.training else 'valid')

        if self.global_step % self.cfg.log_interval == 0 or not self.training:
            self.log_images(imgs, 'imgs')
            self.log_images(edges, 'edges')
            self.log_images(preds, 'preds')

        self.log_dict(loss_log, prog_bar=True)

        return loss