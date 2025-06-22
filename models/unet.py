import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple
from collections.abc import Iterable

from torch.nn import InstanceNorm2d

from models.model import Model
from modules.downsample.conv import ConvDownSample
from modules.norm.layer_norm import LayerNorm
from modules.upsample.pixel_shuffle import PixelShuffleUpSample
from modules.block.conv_next import LocalConvNextV2Block
from modules.norm.grn import GlobalResponseNorm
from utils.load_module import load_module


class UNet(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 drop_path: float = 0.0,
                 activation: str = 'torch.nn.GELU',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 scale_factors: int | List[int] | Tuple[int] = 2,
                 *args,
                 **kwargs,
                 ):
        super(UNet, self).__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        self.hidden_dims = hidden_dims

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.scale_factor = scale_factors.pop(0)
        self.mode = mode.lower()
        assert self.mode in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area', 'nearest-eaxct']

        self.encoder.append(
            nn.Sequential(
                ConvDownSample(
                    in_channels=in_channels,
                    out_channels=embed_dim,
                    scale_factor=self.scale_factor,
                    use_checkpoint=use_checkpoint,
                    activation=activation,
                )
            )
        )

        in_ch = embed_dim
        skip_dims = [ embed_dim ]

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.encoder.append(
                    LocalConvNextV2Block(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

                skip_dims.append(in_ch)

            if i != len(hidden_dims):
                self.encoder.append(
                    ConvDownSample(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=scale_factors[i],
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                    )
                )

                in_ch = out_ch

        self.bottle_neck = nn.Sequential(
            LocalConvNextV2Block(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
            ),
            LocalConvNextV2Block(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                activation=activation,
                drop_path=drop_path,
            ),
        )

        hidden_dims.pop()
        hidden_dims.insert(0, embed_dim)

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            if i != len(hidden_dims):
                self.decoder.append(
                    PixelShuffleUpSample(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=scale_factors[i],
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                    )
                )

                in_ch = out_ch

            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.decoder.append(
                    LocalConvNextV2Block(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

        make_activation = load_module(activation)

        self.out = nn.Sequential(
            PixelShuffleUpSample(
                in_channels=in_ch,
                out_channels=in_ch,
                scale_factor=self.scale_factor,
                use_checkpoint=use_checkpoint,
                activation=activation,
            ),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                padding=1,
                groups=in_ch,
            ),
            LayerNorm(in_ch),
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=1),
            make_activation(),
            GlobalResponseNorm(in_ch * 4),
            nn.Conv2d(in_ch * 4, out_channels, kernel_size=1),
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
            if not isinstance(block, PixelShuffleUpSample):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs)

        # outputs = torch.cat([outputs, skips.pop()], dim=1)

        return self.out(outputs)