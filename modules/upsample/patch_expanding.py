import torch
import torch.nn as nn
from einops import rearrange

from modules.upsample import UpSample
from modules.norm.layer_norm import LayerNorm


class PatchExpanding(UpSample):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.expand = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                (self.scale_factor ** 2) * self.out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.PixelShuffle(int(self.scale_factor))
        )

        self.norm = LayerNorm(self.out_channels)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        return self.norm(x)