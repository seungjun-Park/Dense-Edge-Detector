import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from omegaconf import DictConfig

from models.model import Model
from thirdparty.convnext_v2 import Block, LayerNorm, GRN, trunc_normal_


class LMIES(Model):
    @dataclass
    class Config(Model.Config):
        in_channels: int = 3
        depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
        dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
        drop_path_ratio: float = 0.

    cfg: Config

    def __init__(self,
                 params: DictConfig,
                 ):
        super().__init__(params=params)

    def configure(self):
        super().configure()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(self.cfg.in_channels, self.cfg.dims[0], kernel_size=4, stride=4),
            LayerNorm(self.cfg.dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(self.cfg.dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(self.cfg.dims[i], self.cfg.dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[Block(self.cfg.dims[i], drop_path=self.cfg.drop_path_ratio) for j in range(self.cfg.depths[i])]
            )
            self.stages.append(stage)

        self.out = nn.Sequential(
            nn.GELU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.out(x)


    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.ShortTensor], batch_idx) -> Optional[torch.Tensor]:
        imgs, edges_0, edges_1, edges_2 = batch
        feat_imgs = self(imgs)
        feat_edges_0 = self(edges_0)
        feat_edges_1 = self(edges_1)
        feat_edges_2 = self(edges_2)

        loss, loss_dict = self.loss_fn(feat_imgs, feat_edges_0, feat_edges_1, feat_edges_2, split='train' if self.training else 'valid')
        self.log_dict(loss_dict, prog_bar=False)

        return loss



