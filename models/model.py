import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Union, List, Tuple, Any, Optional

from pytorch_lightning.utilities.parsing import save_hyperparameters

from utils import instantiate_from_config
from losses.loss import Loss


class Model(pl.LightningModule, ABC):
    @dataclass
    class Config:
        lr: float
        log_interval: int
        weight_decay: float = 0.0
        loss_config: DictConfig = None

        ckpt_path: str = None
        ignore_keys: Union[List[str], Tuple[str]] = field(default_factory=tuple)

    def __init__(self,
                 params: DictConfig,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        self.cfg = self.Config(**params)

        self.configure()

        if self.cfg.ckpt_path is not None:
            self.init_from_ckpt(path=self.cfg.ckpt_path, ignore_keys=self.cfg.ignore_keys)

        self.save_hyperparameters()

    def configure(self):
        loss_config = self.cfg.loss_config
        if loss_config is not None:
            self.loss_fn: Loss = instantiate_from_config(loss_config)
        else:
            self.loss_fn = None

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

    @abstractmethod
    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        pass

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def log_images(self, imgs: torch.Tensor, name: str):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/{name}', imgs[0].float(), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        params = list(self.parameters())
        opt_net = torch.optim.AdamW(params,
                                    lr=self.cfg.lr,
                                    weight_decay=self.cfg.weight_decay,
                                    betas=(0.5, 0.9),
                                    )

        schedular_net = torch.optim.lr_scheduler.CosineAnnealingLR(opt_net, T_max=50, eta_min=0)

        opts = [opt_net]
        schs = [schedular_net]

        return opts, schs