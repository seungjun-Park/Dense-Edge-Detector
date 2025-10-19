import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union, List, Tuple, Optional
from collections.abc import Iterable


from models.model import Model
from modules.downsample.conv import ConvDownSample
from modules.upsample.conv import ConvUpSample
from modules.block.res_block import ResidualBlock
from modules.block.attention_block import AttentionBlock
from modules.sequential.cond_sequential import ConditionalSequential

from utils.load_module import load_module


class UNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 channel_mult: Tuple[int] = (1, 2, 4, 8),
                 num_blocks: int = 2,
                 drop_path: float = 0.0,
                 activation: str = 'torch.nn.GELU',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 num_groups: int = 1,
                 num_heads: int = 8,
                 num_head_channels: int = -1,
                 *args,
                 **kwargs,
                 ):
        super(UNet, self).__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        make_activation = load_module(activation)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.embed_dim = embed_dim
        granularity_embed_dim = embed_dim * 4

        self.granularity_embed = nn.Sequential(
            nn.Linear(embed_dim, granularity_embed_dim),
            make_activation(),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1),
        )

        in_ch = embed_dim

        skip_dims = [embed_dim]

        for i, mult in enumerate(channel_mult):
            for j in range(num_blocks):
                self.encoder.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        embed_channels=granularity_embed_dim,
                        out_channels=embed_dim * mult,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                        num_groups=num_groups,
                    ),
                )

                in_ch = embed_dim * mult
                skip_dims.append(in_ch)

            if i != len(channel_mult) - 1:
                self.encoder.append(
                    ConvDownSample(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                    )
                )

                skip_dims.append(in_ch)

        self.bottle_neck = ConditionalSequential(
            ResidualBlock(
                in_channels=in_ch,
                embed_channels=granularity_embed_dim,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
                num_groups=num_groups,
            ),
            AttentionBlock(
                channels=in_ch,
                num_groups=num_groups,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_checkpoint=use_checkpoint
            ),
            ResidualBlock(
                in_channels=in_ch,
                embed_channels=granularity_embed_dim,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
                num_groups=num_groups,
            )
        )

        for i, mult in list(enumerate(channel_mult))[::-1]:
            for j in range(num_blocks):
                self.decoder.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        embed_channels=granularity_embed_dim,
                        out_channels=embed_dim * mult,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                        num_groups=num_groups
                    )
                )
                in_ch = embed_dim * mult

                if i and j == num_blocks:
                    self.decoder.append(
                        ConvUpSample(
                            in_channels=in_ch,
                            mode=mode,
                            use_checkpoint=use_checkpoint,
                        )
                    )

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            make_activation(),
            nn.Conv2d(
                embed_dim,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid()
        )

        if self.loss:
            self.loss.eval()
        self.save_hyperparameters(ignore='loss_config')

    def _granularity_embedding(self, granularity, dim, max_period=10000):
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
            granularity = self._granularity_embedding(granularity, dim=self.embed_dim)
            granularity = self.granularity_embed(granularity)

        skips = [outputs]
        for block in self.encoder:
            outputs = block(outputs, granularity)
            skips.append(outputs)

        outputs = self.bottle_neck(outputs, granularity)

        for block in self.decoder:
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