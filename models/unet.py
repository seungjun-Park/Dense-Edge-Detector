from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Tuple, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig
from timm.models.vision_transformer import DropPath

from models import DefaultModel
from losses.l1lpips import L1LPIPS
from utils import zero_module, granularity_embedding
from utils.checkpoints import checkpoint


class GranularityBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        pass


class GranularityEmbedSequential(nn.Sequential, GranularityBlock):
    def forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, GranularityBlock):
                x = layer(x, granularity)
            else:
                x = layer(x)


class Upsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 ):
        super().__init__()
        out_channels = out_channels if out_channels else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 ):
        super().__init__()

        out_channels = out_channels if out_channels else in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResBlock(GranularityBlock):
    def __init__(self,
                 in_channels: int,
                 embed_channels: int,
                 drop_prob: float = 0.,
                 out_channels: int = None,
                 use_checkpoint: bool = False,
                 num_groups: int = 32,
                 ):
        super().__init__()

        out_channels = out_channels if out_channels else in_channels
        self.use_checkpoint = use_checkpoint
        self.drop_path = DropPath(drop_prob=drop_prob)

        self.emb_layers = nn.Sequential(
            nn.GELU(approximate='tanh'),
            nn.Linear(
                embed_channels,
                out_channels * 2
            )
        )

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.GELU(approximate='tanh'),
            zero_module(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.GELU(approximate='tanh'),
            zero_module(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        args = [x]
        if granularity is not None:
            args.append(granularity)
        return checkpoint(self._forward, tuple(args), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(granularity).type(h.type)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        out_norm, out_rest = self.out_layers[0], self.out_layers[1: ]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)

        return self.skip(x) + self.drop_path(h)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self,
                 n_heads: int
                 ):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels: int,
        num_groups: int = 1,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return checkpoint(self._forward, (x, ), self.parameters(), self.use_checkpoint)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class UNet(DefaultModel):
    @dataclass
    class Config(DefaultModel.Config):
        in_channels: int = 3
        model_channels: int = 32
        out_channels: int = None
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = field(default_factory=lambda : [])
        channels_mult: Tuple[int] = field(default_factory=lambda : (1, 2, 4, 8))
        drop_prob: float = 0.2
        use_checkpoint: bool = True
        num_heads: int = -1,
        num_head_channels: int = -1
        num_groups: int = 32

    cfg: Config

    def __init__(self,
                 params: DictConfig
                 ):
        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.loss_fn = self.loss_fn.eval()

        if self.cfg.num_head_channels == -1:
            assert self.cfg.num_heads != -1

        if self.cfg.num_heads == -1:
            assert self.cfg.num_head_channels != -1

        out_channels = self.cfg.out_channels if self.cfg.out_channels is not None else self.cfg.in_channels

        granularity_embed_dim = self.cfg.model_channels * 4
        self.granularity_embed = nn.Sequential(
            nn.Linear(self.cfg.model_channels, granularity_embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                GranularityEmbedSequential(
                    nn.Conv2d(self.cfg.in_channels, self.cfg.model_channels, kernel_size=3, padding=1)
                )
            ]
        )

        in_ch = self.cfg.model_channels
        skip_dims = [self.cfg.model_channels]
        for i, mult in enumerate(self.cfg.channels_mult):
            for _ in range(self.cfg.num_res_blocks):
                layers = [
                    ResBlock(
                        in_ch,
                        granularity_embed_dim,
                        self.cfg.drop_prob,
                        out_channels=mult * self.cfg.model_channels,
                        use_checkpoint=self.cfg.use_checkpoint,
                        num_groups=self.cfg.num_groups,
                    )
                ]
                in_ch = mult * self.cfg.model_channels
                if i in self.cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            in_ch,
                            use_checkpoint=self.cfg.use_checkpoint,
                            num_groups=self.cfg.num_groups,
                            num_heads=self.cfg.num_heads,
                            num_head_channels=self.cfg.num_head_channels
                        )
                    )
                self.input_blocks.append(GranularityEmbedSequential(*layers))
                skip_dims.append(in_ch)

            if i != len(self.cfg.channels_mult) - 1:
                self.input_blocks.append(
                    Downsample(
                        in_ch,
                    )
                )
                skip_dims.append(in_ch)


        self.middle_blocks = GranularityEmbedSequential(
            ResBlock(
                in_ch,
                granularity_embed_dim,
                self.cfg.drop_prob,
                use_checkpoint=self.cfg.use_checkpoint,
                num_groups=self.cfg.num_groups
            ),
            AttentionBlock(
                in_ch,
                use_checkpoint=self.cfg.use_checkpoint,
                num_groups=self.cfg.num_groups,
                num_heads=self.cfg.num_heads,
                num_head_channels=self.cfg.num_head_channels,
            ),
            ResBlock(
                in_ch,
                granularity_embed_dim,
                self.cfg.drop_prob,
                use_checkpoint=self.cfg.use_checkpoint,
                num_groups=self.cfg.num_groups
            ),
        )

        self.output_blocks = nn.ModuleList([])

        for i, mult in list(enumerate(self.cfg.channels_mult))[::-1]:
            for j in range(self.cfg.num_res_blocks + 1):
                in_ch = in_ch + skip_dims.pop()
                layers = [
                    ResBlock(
                        in_ch,
                        granularity_embed_dim,
                        drop_prob=self.cfg.drop_prob,
                        out_channels=self.cfg.model_channels * mult,
                        use_checkpoint=self.cfg.use_checkpoint,
                        num_groups=self.cfg.num_groups,
                    )
                ]
                in_ch = self.cfg.model_channels * mult
                if i in self.cfg.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            in_ch,
                            use_checkpoint=self.cfg.use_checkpoint,
                            num_groups=self.cfg.num_groups,
                            num_heads=self.cfg.num_heads,
                            num_head_channels=self.cfg.num_head_channels
                        )
                    )

                if i and j == self.cfg.num_res_blocks:
                    layers.append(
                        GranularityEmbedSequential(
                            Upsample(
                                in_ch,
                                in_ch
                            )
                        )
                    )
                self.output_blocks.append(GranularityEmbedSequential(*layers))
                
        self.out = nn.Sequential(
            nn.GroupNorm(self.cfg.num_groups, in_ch),
            nn.GELU(approximate='tanh'),
            zero_module(
                nn.Conv2d(in_ch, self.cfg.out_channels, kernel_size=3, padding=1),
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        skips = []
        granularity = granularity_embedding(granularity, dim=self.cfg.model_channels)
        emb = self.granularity_embed(granularity)
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            skips.append(h)

        h = self.middle_blocks(h)

        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, emb)

        return self.out(h)

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