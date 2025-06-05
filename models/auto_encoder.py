import torch
import torch.nn as nn

from typing import Union, List, Tuple, Type
from collections.abc import Iterable

from models.model import Model
from modules.down import DownBlock
from modules.up import UpBlock
from utils.load_module import  load_module
from utils.params import get_module_params


class AutoEncoder(Model):
    def __init__(self,
                 in_channels: int,
                 embed_dim: int,
                 out_channels: int = None,
                 hidden_dims: Union[List[int], Tuple[int]] = (32, 64, 128, 256),
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 dropout: float = 0.0,
                 drop_path: float = 0.0,
                 conv: str = 'torch.nn.Conv2d',
                 activation: str = 'torch.nn.GELU',
                 norm: str = None,
                 num_groups: int = None,
                 pooling: str = 'torch.nn.Conv2d',
                 mode: str = 'nearest',
                 use_checkpoint: bool = True,
                 *args,
                 **kwargs,):
        super(AutoEncoder, self).__init__(*args, **kwargs)

        out_channels = out_channels if out_channels is not None else in_channels

        self.hidden_dims = hidden_dims

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        make_activation: Type[nn.Module] = load_module(activation)
        make_norm: Type[nn.Module] = load_module(norm)
        make_conv: Type[nn.Module] = load_module(conv)

        self.encoder.append(
            nn.Sequential(
                make_conv(
                    in_channels,
                    embed_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        )

        in_ch = embed_dim

        for i, out_ch in enumerate(hidden_dims):
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.encoder.append(
                    DepthWiseSeperableResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        conv=conv,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        dropout=dropout,
                        drop_path=drop_path,
                        use_checkpoint=use_checkpoint,
                        use_conv=True,
                    )
                )

                in_ch = out_ch

            if i != len(hidden_dims) - 1:
                self.encoder.append(
                    DownBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        pooling=pooling,
                        use_checkpoint=use_checkpoint,
                    )
                )

        for i, out_ch in list(enumerate(hidden_dims))[::-1]:
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.decoder.append(
                    DepthWiseSeperableResidualBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        conv=conv,
                        activation=activation,
                        norm=norm,
                        num_groups=num_groups,
                        dropout=dropout,
                        drop_path=drop_path,
                        use_checkpoint=use_checkpoint,
                        use_conv=True,
                    )
                )

                in_ch = out_ch

            if i != 0:
                self.decoder.append(
                    UpBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        scale_factor=2,
                        conv=conv,
                        mode=mode,
                        use_checkpoint=use_checkpoint,
                    )
                )

        get_norm_params = get_module_params(make_norm)

        self.decoder.append(
            nn.Sequential(
                make_norm(**get_norm_params(in_ch, num_groups=num_groups)),
                make_activation,
                make_conv(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.Sigmoid(),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs
        for block in self.encoder:
            outputs = block(outputs)
        for block in self.decoder:
            outputs = block(outputs)

        return outputs