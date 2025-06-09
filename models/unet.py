import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from typing import Union, List, Tuple, Type
from collections.abc import Iterable

from models.model import Model
from modules.down import DownBlock
from modules.up import UpBlock
from modules.res_block import DepthwiseSeperableResidualBlock
from modules.attention import FlashAttentionBlock
from modules.norm import LayerNorm
from utils.load_module import load_module
from utils import zero_module


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
                 num_heads: int = 8,
                 num_head_channels: int = None,
                 softmax_scale: float = None,
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
                nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=self.scale_factor,
                    stride=self.scale_factor,
                ),
                LayerNorm(embed_dim, data_format='channels_first')
            )
        )

        in_ch = embed_dim
        skip_dims = []

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.encoder.append(
                    DepthwiseSeperableResidualBlock(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

                skip_dims.append(in_ch)

            if i != len(hidden_dims):
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=scale_factors[i],
                        use_checkpoint=use_checkpoint,
                    )
                )

                in_ch = out_ch

        self.bottle_neck = nn.Sequential(
            FlashAttentionBlock(
                in_channels=in_ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                softmax_scale=softmax_scale,
                activation=activation,
                drop_path=drop_path,
            ),
        )

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            if i != len(hidden_dims):
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        scale_factor=scale_factors[i],
                        mode=mode,
                        use_checkpoint=use_checkpoint,
                    )
                )

                in_ch = out_ch

            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.decoder.append(
                    DepthwiseSeperableResidualBlock(
                        in_channels=in_ch + skip_dims.pop(),
                        out_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

        make_activation = load_module(activation)

        self.conv = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
        )

        self.out = nn.Sequential(
            nn.LayerNorm(in_ch, eps=1e-6),
            nn.Linear(in_ch, in_ch * 4),
            make_activation(),
            nn.Linear(in_ch * 4, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        skips = []
        for block in self.encoder:
            outputs = block(outputs)
            if not isinstance(block, DownBlock):
                skips.append(outputs)

        outputs = self.bottle_neck(outputs)

        for block in self.decoder:
            if not isinstance(block, UpBlock):
                outputs = torch.cat([outputs, skips.pop()], dim=1)
            outputs = block(outputs)

        outputs = F.interpolate(outputs, scale_factor=self.scale_factor, mode=self.mode)
        outputs = self.conv(outputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.out(outputs)
        outputs = outputs.permute(0, 3, 1, 2)

        return outputs