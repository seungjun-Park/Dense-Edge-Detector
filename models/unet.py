import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from collections.abc import Iterable


from models.model import Model
from modules.downsample.conv import ConvDownSample
from modules.norm.layer_norm import LayerNorm
from modules.upsample.conv import ConvUpSample
from modules.block.attention import FlashAttentionBlock
from modules.block.res_block import ResidualBlock
from utils.load_module import load_module
from modules.block.squeeze_excitation import SEBlock


class UNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 drop_path: float = 0.0,
                 activation: str = 'torch.nn.GELU',
                 num_heads: int = 8,
                 num_head_channels: int = None,
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 scale_factors: int | List[int] | Tuple[int] = 2,
                 *args,
                 **kwargs,
                 ):
        super(UNet, self).__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    padding=1,
                ),
                LayerNorm(embed_dim)
            )
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

        self.bottle_neck = nn.Sequential(
            ResidualBlock(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
            ),
            FlashAttentionBlock(
                in_channels=in_ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                activation=activation,
                drop_path=drop_path,
            ),
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
                    )
                )

        self.out = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.save_hyperparameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs

        skips = []
        for block in self.encoder:
            outputs = block(outputs)
            if not isinstance(block, ConvDownSample):
                skips.append(outputs)

        outputs = self.bottle_neck(outputs)

        for block in self.decoder:
            if not isinstance(block, ConvUpSample):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs)

        return self.out(outputs)