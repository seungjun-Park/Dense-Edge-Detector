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
from utils import zero_module, granularity_embedding
from modules.block.res_block import ResidualBlock
from modules.block.downsample import DownSample
from modules.block.upsample import Upsample
from modules.sequential.cond_sequential import ConditionalSequential


class UNet(DefaultModel):
    @dataclass
    class Config(DefaultModel.Config):
        in_channels: int = 3
        model_channels: int = 32
        out_channels: int = None
        num_res_blocks: int = 2,
        channels_mult: Tuple[int] = field(default_factory=lambda : (1, 2, 4, 8))
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

        granularity_embed_dim = self.cfg.model_channels * 4
        self.granularity_embed = nn.Sequential(
            nn.Linear(self.cfg.model_channels, granularity_embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.embed = nn.Conv2d(self.cfg.in_channels, self.cfg.model_channels, kernel_size=7, padding=3)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        in_ch = self.cfg.model_channels
        skip_dims = [self.cfg.model_channels]

        for i, mult in enumerate(self.cfg.channels_mult):
            blocks = []
            for _ in range(self.cfg.num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        embed_channels=granularity_embed_dim,
                        drop_prob=self.cfg.drop_prob,
                        use_checkpoint=self.cfg.use_checkpoint
                    )
                )
            self.encoder.append(ConditionalSequential(*blocks))
            skip_dims.append(in_ch)

            self.encoder.append(
                DownSample(
                    in_ch,
                    self.cfg.model_channels * mult
                )
            )
            in_ch = self.cfg.model_channels * mult

        self.bottle_neck = ConditionalSequential(
            ResidualBlock(
                in_channels=in_ch,
                embed_channels=granularity_embed_dim,
                drop_prob=self.cfg.drop_prob,
                use_checkpoint=self.cfg.use_checkpoint
            ),
            ResidualBlock(
                in_channels=in_ch,
                embed_channels=granularity_embed_dim,
                drop_prob=self.cfg.drop_prob,
                use_checkpoint=self.cfg.use_checkpoint
            )
        )

        for i, mult in list(enumerate(self.cfg.channels_mult))[::-1]:
            self.decoder.append(
                Upsample(
                    in_channels=in_ch,
                    out_channels=self.cfg.model_channels * mult
                )
            )

            in_ch = self.cfg.model_channels * mult

            blocks = []

            in_ch = in_ch + skip_dims.pop()
            for j in range(self.cfg.num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        embed_channels=granularity_embed_dim,
                        drop_prob=self.cfg.drop_prob,
                        use_checkpoint=self.cfg.use_checkpoint
                    )
                )

            self.decoder.append(ConditionalSequential(*blocks))

        self.out = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.GELU(approximate='tanh'),
            zero_module(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        skips = []
        granularity = granularity_embedding(granularity, dim=self.cfg.model_channels)
        emb = self.granularity_embed(granularity)

        x = self.embed(x)

        for module in self.encoder:
            x = module(x, emb)
            if not isinstance(module, DownSample):
                skips.append(x)

        x = self.bottle_neck(x, emb)

        for module in self.decoder:
            if not isinstance(module, Upsample):
                x = torch.cat([x, skips.pop()], dim=1)
            x = module(x, emb)

        return self.out(x)

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