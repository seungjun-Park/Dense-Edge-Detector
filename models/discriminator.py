from typing import Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import sequence

from models.model import Model
from modules.norm.act_norm import ActNorm


class Discriminator(Model):
    def __init__(self,
                 ndf: int = 64,
                 n_layers: int = 3,
                 use_actnorm: bool = True,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        if not use_actnorm:
            norm_layer = nn.BatchNorm2d

        else:
            norm_layer = ActNorm

        use_bias = (norm_layer != nn.BatchNorm2d)

        kw = 4
        padw = 1

        self.img_embed = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        )

        self.edge_embed = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        )

        nf_mult = 1
        nf_mult_prev = 1

        sequence = [nn.Conv2d(ndf * 2, ndf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.main = nn.Sequential(*sequence)
        self.out = nn.Sequential(
            nn.Linear(ndf * nf_mult * 7 * 7, 1),
            nn.Sigmoid(),
        )
    def forward(self, imgs: torch.Tensor, edges: torch.Tensor) -> Any:
        imgs = self.img_embed(imgs)
        edges = self.edge_embed(edges)
        h = self.main(torch.cat([imgs, edges], dim=1))
        h = F.adaptive_avg_pool2d(h, (7, 7)).flatten(start_dim=1)
        return self.out(h)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        pair0, pair1, labels = batch
        d0 = self(pair0[0], pair0[1])
        d1 = self(pair1[0], pair1[1])

        loss, loss_log = self.loss(d0, d1, labels, split='train' if self.training else 'valid')

        self.log_dict(loss_log, prog_bar=True)

        return loss