import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Tuple, Optional
from collections.abc import Iterable
from dataclasses import dataclass, field

from omegaconf import DictConfig
from timm.layers.grn import GlobalResponseNorm

from models.model import Model
from modules.sequential.cond_sequential import ConditionalSequential

from modules.norm.layer_norm import LayerNorm2d
from losses.l1lpips import L1LPIPS


class Block(nn.Module):
    def __init__(self,
                 in_channels: int,
                 embed_channels: int,
                 dropout: float = 0.3,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.embed = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_channels, in_channels * 2)
        )

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels , in_channels * 4)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * in_channels, channels_last=True)
        self.pwconv2 = nn.Linear(in_channels * 4, in_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        h = self.dwconv(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)

        if granularity is not None:
            granularity = self.embed(granularity).type(h.dtype)
            granularity = granularity[:, None, None, :]
            scale, shift = granularity.chunk(2, dim=-1)
            h = h * (1 + scale) + shift

        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.dropout(h)
        h = self.pwconv2(h)

        h = h.permute(0, 3, 1, 2)

        return h



class ShallowNet(Model):
    @dataclass
    class Config(Model.Config):
        in_channels: int = 3
        embed_dim: int = 32
        out_channels: int = None
        use_checkpoint: bool = True
        dropout: float = 0.3

    cfg: Config

    def __init__(self,
                 params: DictConfig
                 ):
        super(ShallowNet, self).__init__(params=params)

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

        self.embed = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.in_channels, kernel_size=7, padding=3, groups=self.cfg.in_channels),
            nn.Conv2d(self.cfg.in_channels, self.cfg.embed_dim, kernel_size=1),
            LayerNorm2d(self.cfg.embed_dim),
        )

        in_ch = self.cfg.embed_dim

        self.block = ConditionalSequential(
            Block(
                in_ch,
                granularity_embed_dim,
                dropout=self.cfg.dropout
            )
        )

        self.out = nn.Sequential(
            LayerNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch),
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
        if granularity.ndim == 1:
            granularity = granularity[:, None].float()
        elif granularity.ndim == 2:
            granularity = granularity.float()
        else:
            raise NotImplementedError(f"granularity ndim should be lower than 3. ndim < 3")

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=granularity.device)
        args = granularity * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, inputs: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        outputs = self.embed(inputs)
        if granularity is not None:
            granularity = self.fourier_embedding(granularity, dim=self.cfg.embed_dim)
            granularity = self.granularity_embed(granularity)

        return self.out(self.block(outputs, granularity))

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges, labels = batch
        with torch.no_grad():
            if isinstance(self.loss_fn, L1LPIPS) and edges is not None:
                labels = self.loss_fn.gnet(imgs, edges)

        preds = self(imgs, labels)

        loss, loss_log = self.loss_fn(imgs, preds, edges, labels, split='train' if self.training else 'valid')

        if self.global_step % self.cfg.log_interval == 0 or not self.training:
            self.log_images(imgs, 'imgs')
            self.log_images(edges, 'edges')
            self.log_images(preds, 'preds')

        self.log_dict(loss_log, prog_bar=True)

        return loss