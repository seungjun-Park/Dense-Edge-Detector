import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple, Any, Optional

from models.model import Model
from torchvision import models
from modules.norm.layer_norm import LayerNorm
from modules.block.film import FiLM


class GranularityPredictor(Model):
    def __init__(self,
                 use_dropout: bool = True,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.net = VGG16(pretrained=True, requires_grad=False)
        self.films = nn.ModuleList()
        self.lins_align = nn.ModuleList()
        self.lins_raw = nn.ModuleList()
        self.lins_shift = nn.ModuleList()

        for i in range(5):
            ch = 64 * min(2 ** i, 8)
            self.films.append(
                FiLM(
                    in_channels=ch,
                )
            )
            self.lins_align.append(
                NetLinLayer(ch, use_dropout=use_dropout)
            )
            self.lins_raw.append(
                NetLinLayer(ch, use_dropout=use_dropout)
            )
            self.lins_shift.append(
                NetLinLayer(ch, use_dropout=use_dropout)
            )

        self.granularity = Dist2Granularity()

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, return_val: bool = False) -> torch.Tensor:
        f_imgs = self.net(imgs)
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)
        f_edges = self.net(edges)
        f_bars = []
        for i in range(5):
            f_bars.append(self.films[i](f_imgs[i], f_edges[i]))

        val_aligns = 0
        val_raw = 0
        val_shift = 0

        for i in range(5):
            f_img = normalize_tensor(f_imgs[i])
            f_edge = normalize_tensor(f_edges[i])
            f_bar = normalize_tensor(f_bars[i])

            d_align = (f_bar - f_edge) ** 2
            d_raw = (f_img - f_edge) ** 2
            d_shift = (f_bar - f_img) ** 2

            res_align = spatial_average(self.lins_align[i](d_align), keepdim=True)
            res_raw = spatial_average(self.lins_raw[i](d_raw), keepdim=True)
            res_shift = spatial_average(self.lins_shift[i](d_shift), keepdim=True)

            val_aligns += res_align
            val_raw += res_raw
            val_shift += res_shift

        granularity = self.granularity(val_aligns, val_raw, val_shift)

        if return_val:
            return granularity.reshape((-1, 1)), val_aligns, val_raw, val_shift

        return granularity.reshape((-1, 1))

    def get_features(self, imgs: torch.Tensor, edges: torch.Tensor) -> List[torch.Tensor]:
        f_imgs = self.net(imgs)
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)
        f_edges = self.net(edges)
        f_bars = []
        for i in range(5):
            f_bars.append(self.films[i](f_imgs[i], f_edges[i]))

        return f_imgs, f_edges, f_bars

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        images, edges, granularity = batch
        preds, val_align, val_raw, val_shift = self(images, edges, return_val=True)

        loss, loss_log = self.loss(preds, granularity, val_align, val_raw, val_shift, split='train' if self.training else 'valid')

        self.log_dict(loss_log, prog_bar=True)

        return loss


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True)).clamp_min(eps)
    return in_feat / norm_factor


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        return vgg_outputs


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self,
                 chn_in: int,
                 chn_out: int = 1,
                 use_dropout: bool = False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Dist2Granularity(nn.Module):
    def __init__(self,
                 chn_mid: int =32,
                 use_sigmoid: bool = True,
                 ):
        super().__init__()

        layers = [nn.Conv2d(6, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2, True), ]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True), ]
        if use_sigmoid:
            layers += [nn.Sigmoid(), ]
        self.model = nn.Sequential(*layers)

    def forward(self, d_align: torch.Tensor, d_raw: torch.Tensor, d_shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return self.model.forward(torch.cat((d_align, d_raw, d_align - d_raw, d_align/(d_raw+eps),d_raw/(d_align+eps), d_shift), dim=1))
