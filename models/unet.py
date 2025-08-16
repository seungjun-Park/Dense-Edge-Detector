import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple, Optional
from collections.abc import Iterable


from models.model import Model
from modules.downsample.conv import ConvDownSample
from modules.norm.layer_norm import LayerNorm
from modules.upsample.conv import ConvUpSample
from modules.block.res_block import ResidualBlock
from modules.embed.gaussian import GaussianFourierEmbedding

from utils.load_module import load_module


class UNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 drop_path: float = 0.0,
                 activation: str = 'torch.nn.GELU',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 scale_factors: int | List[int] | Tuple[int] = 2,
                 num_groups: int = 1,
                 *args,
                 **kwargs,
                 ):
        super(UNet, self).__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        make_activation = load_module(activation)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        granularity_embed_dim = embed_dim * 4

        self.granularity_embed = nn.Sequential(
            GaussianFourierEmbedding(embed_dim, learnable=True),
            nn.Linear(embed_dim, granularity_embed_dim),
            make_activation(),
            nn.Linear(granularity_embed_dim, granularity_embed_dim),
        )

        self.embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            padding=1,
        )

        in_ch = embed_dim

        skip_dims = []

        for i, sf in enumerate(scale_factors):
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.encoder.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                        num_groups=num_groups,
                    )
                )
                skip_dims.append(in_ch)

            self.encoder.append(
                ConvDownSample(
                    in_channels=in_ch,
                    scale_factor=sf,
                    use_checkpoint=use_checkpoint,
                )
            )

            in_ch = int(in_ch * sf)

        self.bottle_neck = nn.ModuleList([
            ResidualBlock(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
                num_groups=num_groups,
            ),
            ResidualBlock(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
                num_groups=num_groups,
            )]
        )

        for i, sf in list(enumerate(scale_factors))[::-1]:
            self.decoder.append(
                ConvUpSample(
                    in_channels=in_ch,
                    scale_factor=sf,
                    mode=mode,
                    use_checkpoint=use_checkpoint,
                )
            )

            in_ch = int(in_ch // sf)

            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.decoder.append(
                    ResidualBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                        num_groups=num_groups
                    )
                )

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            make_activation(),
            nn.Conv2d(
                in_ch,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, inputs: torch.Tensor, granularity: torch.Tensor) -> torch.Tensor:
        outputs = self.embed(inputs)
        granularity = self.granularity_embed(granularity)

        skips = []
        for block in self.encoder:
            outputs = block(outputs, granularity)
            if not isinstance(block, ConvDownSample):
                skips.append(outputs)

        for block in self.bottle_neck:
            outputs = block(outputs, granularity)

        for block in self.decoder:
            if not isinstance(block, ConvUpSample):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs, granularity)

        return self.out(outputs)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        inputs, targets, granularity = batch
        outputs = self(inputs, granularity)

        loss, loss_log = self.loss(inputs, targets, outputs, granularity, split='train' if self.training else 'valid')

        if self.global_step % self.log_interval == 0:
            self.log_images(inputs, targets, outputs)

        self.log_dict(loss_log, prog_bar=True)

        return loss