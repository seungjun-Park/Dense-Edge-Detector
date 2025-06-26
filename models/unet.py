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
from modules.block.conv_next import ConvNextV2Block
from utils.load_module import load_module
from modules.block.squeeze_excitation import SEBlock


class UNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 block_types: str | List[str] = [],
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

        assert len(block_types) == len(scale_factors)

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
            encoder = []
            make_block = load_module(block_types[i])
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                encoder.append(
                    make_block(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

            self.encoder.append(*encoder)

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
            ConvNextV2Block(
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
            ConvNextV2Block(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
            ),
        )

        for i, sf in list(enumerate(scale_factors))[::-1]:
            make_block = load_module(block_types[i])

            self.decoder.append(
                ConvUpSample(
                    in_channels=in_ch,
                    scale_factor=sf,
                    mode=mode,
                    use_checkpoint=use_checkpoint,
                )
            )

            in_ch = int(in_ch * sf)

            decoder = []

            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                decoder.append(
                    make_block(
                        in_channels=in_ch + skip_dims.pop() if j == 0 else in_ch,
                        out_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

            self.decoder.append(*decoder)

        self.out = nn.Sequential(
            LayerNorm(in_ch),
            SEBlock(
                in_channels=in_ch,
                embed_ratio=2,
                activation=activation,
            ),
            nn.Conv2d(
                in_ch,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs

        skips = []
        for block in self.encoder:
            if isinstance(block, ConvDownSample):
                skips.append(outputs)
            outputs = block(outputs)

        outputs = self.bottle_neck(outputs)

        for block in self.decoder:
            if not isinstance(block, ConvUpSample):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs)

        return self.out(outputs)