from typing import Tuple, Optional, Any, Union, Callable, List, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from utils import instantiate_from_config
from losses.loss import Loss


def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return in_tens.mean([2, 3], keepdim=keepdim)


class Adaptor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 reduction_ratio: float = 2,
                 drop_prob: float = 0.1,
                 ):
        super().__init__()

        hidden_dim = int(in_channels // reduction_ratio)

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None ,None])
        self.register_buffer('scale', torch.Tensor([-.458, -.448, -.450])[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 drop_prob: float = 0.1,
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LPIEPS(pl.LightningModule):
    def __init__(self,
                 lr: float = 1e-4,
                 log_interval: int = 100,
                 weight_decay: float = 0.0,
                 loss_config: DictConfig = None,
                 ckpt_path: str = None,
                 ignore_keys: Tuple[str] = (),
                 ):

        super().__init__()

        self.lr = lr
        self.log_interval = log_interval
        self.weight_decay = weight_decay

        self.backbone = timm.create_model('vgg16_bn', pretrained=True, features_only=True)

        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_channels = self.backbone.feature_info.channels()

        self.scaling_layer = ScalingLayer()
        self.lins = nn.ModuleList()
        self.adaptors = nn.ModuleList()

        for c in feature_channels:
            self.lins.append(
                NetLinLayer(c)
            )

            self.adaptors.append(
                Adaptor(c)
            )

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

        if loss_config is not None:
            self.loss_fn: Loss = instantiate_from_config(loss_config)
        else:
            self.loss_fn = None

        self.automatic_optimization = False

        self.save_hyperparameters(ignore=['loss_config'])

    def init_from_ckpt(self, path, ignore_keys=list()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path}")

    def forward(self, imgs: torch.Tensor, edges: torch.Tensor, return_feats: bool = False) -> torch.Tensor:
        b = imgs.shape[0]
        if edges.shape[1] == 1:
            edges = edges.repeat(1, 3, 1, 1)

        imgs = self.scaling_layer(imgs)
        edges = self.scaling_layer(edges)

        feats_imgs = self.backbone(imgs)
        feats_edges = self.backbone(edges)

        val = 0
        adaptors_feats = []
        for adaptor, lin, feat_imgs, feat_edges in zip(self.adaptors, self.lins, feats_imgs, feats_edges):
            feat_imgs = adaptor(feat_imgs)
            adaptors_feats.append(feat_imgs)
            feat_imgs = feat_imgs.detach()
            diff = (normalize_tensor(feat_imgs) - normalize_tensor(feat_edges)) ** 2
            res = spatial_average(lin(diff))
            val += res

        if return_feats:
            return  val.reshape(b), adaptors_feats, feats_edges

        return val.reshape(b)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        opt_adaptors, opt_lins = self.optimizers()

        imgs, edges_0, edges_1, edges_2 = batch

        d_high, adaptors_feats, feats_edges = self(imgs, edges_0, True)
        d_mid = self(imgs, edges_1)
        d_poor = self(imgs, edges_2)

        loss_adaptor = 0
        for adaptor_feats, feat_edges in zip(adaptors_feats, feats_edges):
            loss_adaptor += F.mse_loss(adaptor_feats, feat_edges)

        opt_adaptors.zero_grad()
        self.manual_backward(loss_adaptor)
        opt_adaptors.step()

        split = 'train' if self.training else 'valid'

        loss, loss_log = self.loss_fn(d_high, d_mid, d_poor, split=split)

        opt_lins.zero_grad()
        self.manual_backward(loss)
        opt_lins.step()

        self.log_dict(loss_log)
        self.log(f'{split}/loss_adaptor', loss_adaptor.clone().detach())

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        opt_adaptors, opt_lins = self.optimizers()

        imgs, edges_0, edges_1, edges_2 = batch

        d_high, adaptors_feats, feats_edges = self(imgs, edges_0, True)
        d_mid = self(imgs, edges_1)
        d_poor = self(imgs, edges_2)

        loss_adaptor = 0
        for adaptor_feats, feat_edges in zip(adaptors_feats, feats_edges):
            loss_adaptor += F.mse_loss(adaptor_feats, feat_edges)

        split = 'train' if self.training else 'valid'

        loss, loss_log = self.loss_fn(d_high, d_mid, d_poor, split=split)

        self.log_dict(loss_log)
        self.log(f'{split}/loss_adaptor', loss_adaptor.clone().detach())

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'lins' in name:
                    param.clamp_(min=1e-6)

    def on_train_epoch_end(self) -> None:
        sch_adaptors, sch_lins = self.lr_schedulers()
        sch_adaptors.step()
        sch_lins.step()

    def configure_optimizers(self) -> Any:
        opt_adaptors = torch.optim.AdamW(list(self.adaptors.parameters()),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay,
                                         betas=(0.5, 0.99),
                                         )

        opt_lins = torch.optim.AdamW(list(self.lins.parameters()),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     betas=(0.5, 0.99),
                                     )

        schedular_adaptors = torch.optim.lr_scheduler.CosineAnnealingLR(opt_adaptors, T_max=50, eta_min=0)
        schedular_lins = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lins, T_max=50, eta_min=0)

        opts = [opt_adaptors, opt_lins]
        schs = [schedular_adaptors, schedular_lins]

        return opts, schs