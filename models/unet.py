import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Tuple, Optional
from collections.abc import Iterable
from dataclasses import dataclass, field

from omegaconf import DictConfig

from models.model import Model
from modules.downsample.conv import ConvDownSample
from modules.upsample.conv import ConvUpSample
from modules.block.res_block import ResidualBlock
from modules.sequential.cond_sequential import ConditionalSequential

from utils.load_module import load_module
from modules.norm.layer_norm import LayerNorm2d


class UNet(Model):
    @dataclass
    class Config(Model.Config):
        in_channels: int = 3
        embed_dim: int = 32
        out_channels: int = None
        channel_mult: Tuple[int] = field(default_factory=lambda : (1, 2, 4, 8))
        depths: Union[List[int]] = field(default_factory=lambda : (2, 2, 2, 2))
        drop_path: float = 0.0
        mode: str = 'nearest'
        use_checkpoint: bool = True

    cfg: Config

    def __init__(self,
                 params: DictConfig
                 ):
        super(UNet, self).__init__(params=params)

    def configure(self):
        out_channels = self.cfg.out_channels if self.cfg.out_channels is not None else self.cfg.in_channels

        granularity_embed_dim = self.cfg.embed_dim * 4

        self.granularity_embed = nn.Sequential(
            nn.Linear(self.cfg.embed_dim, granularity_embed_dim),
            nn.GELU(),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.embed = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.in_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.Conv2d(self.cfg.in_channels, self.cfg.embed_dim, kernel_size=1),
            LayerNorm2d(self.cfg.embed_dim),
        )

        in_ch = self.cfg.embed_dim

        skip_dims = [self.cfg.embed_dim]

        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for i, mult in enumerate(self.cfg.channel_mult):
            self.encoder_stages.append(
                ConditionalSequential(
                    *[
                        ResidualBlock(
                            in_ch,
                            embed_channels=granularity_embed_dim,
                            drop_path=self.cfg.drop_path
                        )
                        for j in range(self.cfg.depths[i])
                    ]
                )
            )

            skip_dims.append(in_ch)

            if i != len(self.cfg.channel_mult) - 1:
                self.encoder_stages.append(
                    ConvDownSample(
                        in_ch,
                        self.cfg.embed_dim * mult
                    )
                )

                in_ch = self.cfg.embed_dim * mult

        for i, mult in list(enumerate(self.cfg.channel_mult))[::-1]:
            in_ch = in_ch + skip_dims.pop()
            self.decoder_stages.append(
                ConditionalSequential(
                    *[
                        ResidualBlock(
                            in_ch,
                            embed_channels=granularity_embed_dim,
                            drop_path=self.cfg.drop_path
                        )
                        for j in range(self.cfg.depths[i])
                    ]
                )
            )

            if i != 0:
                self.decoder_stages.append(
                    ConvUpSample(
                        in_ch,
                        self.cfg.embed_dim * mult,
                        mode=self.cfg.mode
                    )
                )

                in_ch = self.cfg.embed_dim * mult

        self.out = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False, groups=in_ch),
            nn.GELU(),
            nn.Conv2d(in_ch, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def fourier_embedding(self, granularity, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=granularity.device)
        args = granularity[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, inputs: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        outputs = self.embed(inputs)
        if granularity is not None:
            granularity = self.fourier_embedding(granularity, dim=self.embed_dim)
            granularity = self.granularity_embed(granularity)

        skips = [outputs]
        for block in self.encoder_stages:
            outputs = block(outputs, granularity)
            if not isinstance(block, ConvDownSample):
                skips.append(outputs)

        for block in self.decoder_stages:
            if not isinstance(block, ConvUpSample):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs, granularity)

        return self.out(outputs)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges0, edges1, edges2 = batch
        preds0 = self(imgs, torch.full((imgs.shape[0],), 1.0).to(imgs.device))
        preds1 = self(imgs, torch.full((imgs.shape[0],), 0.5).to(imgs.device))
        preds2 = self(imgs, torch.full((imgs.shape[0],), 0.0).to(imgs.device))

        loss, loss_log = self.loss(imgs, edges0, edges1, edges2, preds0, preds1, preds2, self.global_step, split='train' if self.training else 'valid')

        if self.global_step % self.log_interval == 0 or not self.training:
            self.log_images(imgs, 'imgs')
            self.log_images(edges0, 'edges0')
            self.log_images(preds0, 'preds0')
            self.log_images(edges1, 'edges1')
            self.log_images(preds1, 'preds1')
            self.log_images(edges2, 'edges2')
            self.log_images(preds2, 'preds2')

        self.log_dict(loss_log, prog_bar=True)

        return loss