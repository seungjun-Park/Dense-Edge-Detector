import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from typing import List, Union, Tuple, Optional
from kornia.color import rgb_to_lab

from models.model import Model
from modules.block import Block
from modules.sequential.cond_sequential import ConditionalSequential
from utils.load_module import load_module


class ConvBlock(Block):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 activation: str,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        make_activation = load_module(activation)

        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False)

        self.out = nn.Sequential(
            nn.GroupNorm(1, embed_dim),
            make_activation()
        )


    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        if h.shape[1] == x.shape[1]:
            h = h + x

        return self.out(h)



class ShallowConvNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 activation: str = 'torch.nn.GELU',
                 num_blocks: int = 1,
                 last_embed_dim: int = 8,
                 use_checkpoint: bool = True,
                 *args,
                 **kwargs,
                 ):

        super().__init__(*args, **kwargs)

        make_activation = load_module(activation)

        self.block = []
        self.block.append(
            ConvBlock(
                in_channels=in_channels,
                embed_dim=embed_dim,
                activation=activation,
                use_checkpoint=use_checkpoint,
            )
        )

        for i in range(num_blocks):
            self.block.append(
                ConvBlock(
                    in_channels=embed_dim,
                    embed_dim=embed_dim,
                    activation=activation,
                    use_checkpoint=use_checkpoint,
                )
            )

        in_ch = embed_dim

        while in_ch > last_embed_dim:
            assert in_ch % 2 == 0
            self.block.append(
                ConvBlock(
                    in_channels=in_ch,
                    embed_dim=in_ch // 2,
                    activation=activation,
                    use_checkpoint=use_checkpoint,
                )
            )
            in_ch //= 2

        self.block.append(
            nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)
        )

        self.block = nn.Sequential(*self.block)

    def make_high_frequency_data(self, imgs: torch.Tensor, ratio: float = 0.2):
        b, c, h, w = imgs.shape
        frequency = fft.fftshift(fft.fft2(imgs))
        half_h, half_w = int(h // 2), int(w // 2)
        ratio_h, ratio_w = int(half_h * ratio), int(half_w * ratio)
        mask = torch.ones_like(frequency)
        mask[:, :, half_h - ratio_h: half_h + ratio_h , half_w - ratio_w: half_w + ratio_w] = 0

        frequency *= mask

        high = fft.ifft2(fft.ifftshift(frequency)).abs()
        high = self.cond(high)

        return high

    def forward(self, imgs: torch.Tensor, granularity: torch.Tensor = None) -> torch.Tensor:
        # high = self.make_high_frequency_data(imgs)
        return F.sigmoid(self.block(imgs))


    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        inputs, targets, granularity = batch
        outputs = self(inputs, granularity)

        loss, loss_log = self.loss(inputs, targets, outputs, granularity, split='train' if self.training else 'valid')

        if self.global_step % self.log_interval == 0:
            self.log_images(inputs, targets, outputs)

        self.log_dict(loss_log, prog_bar=True)

        return loss