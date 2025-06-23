import torch
import torch.nn as nn

from modules.downsample import DownSample
from modules.norm.layer_norm import LayerNorm


class PatchMerging(DownSample):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.reduction = nn.Sequential(
            LayerNorm((self.scale_factor ** 2) * self.in_channels),
            nn.Conv2d(
                (self.scale_factor ** 2) * self.in_channels,
                self.out_channels,
                kernel_size=1,
                bias=False,
            )
        )


    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scaled_h = h // self.scale_factor
        scaled_w = w // self.scale_factor

        x = x.reshape(b, c, self.scale_factor,scaled_h, self.scale_factor, scaled_w)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, -1, scaled_h, scaled_w)

        return self.reduction(x)