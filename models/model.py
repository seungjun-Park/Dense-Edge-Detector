import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
from collections import abc
from abc import ABC, abstractmethod

from typing import Union, List, Tuple, Any, Optional
from utils import instantiate_from_config
from losses.loss import Loss


class Model(pl.LightningModule, ABC):
    def __init__(self,
                 loss_config: DictConfig = None,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 lr_decay_epoch: int = 100,
                 log_interval: int = 100,
                 ckpt_path: str = None,
                 ignore_keys: Union[List[str], Tuple[str]] = (),
                 noise_prob: float = 0.,
                 max_noise_ratio: float = 0.,
                 *args,
                 **ignored_kwargs,
                 ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.log_interval = log_interval
        self.max_noise_ratio = max_noise_ratio
        self.noise_prob = noise_prob

        if loss_config is not None:
            self.loss: Loss = instantiate_from_config(loss_config).eval()
        else:
            self.loss = None

        if ckpt_path is not None:
            self.init_from_ckpt(path=ckpt_path, ignore_keys=ignore_keys)

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
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    def add_noise(self, inputs: torch.Tensor) -> torch.Tensor:
        if random.random() < self.noise_prob:
            noise = torch.randn(inputs.shape).to(inputs.device)
            # prob = min(self.add_noise_ratio, random.random())
            inputs = inputs + self.max_noise_ratio * (self.current_epoch / self.trainer.max_epochs) * noise

        return inputs

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        inputs, targets, _ = batch
        # noised_inputs = self.add_noise(inputs)
        outputs = self(inputs)

        loss, loss_log = self.loss(inputs, targets, outputs, split='train' if self.training else 'valid')

        if self.global_step % self.log_interval == 0:
            self.log_images(inputs, targets, outputs)

        self.log_dict(loss_log, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    @torch.no_grad()
    def log_images(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor):
        prefix = 'train' if self.training else 'val'
        tb = self.logger.experiment
        tb.add_image(f'{prefix}/inputs', inputs[0].float(), self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/targets', targets[0].float(), self.global_step, dataformats='CHW')
        tb.add_image(f'{prefix}/outputs', outputs[0].float(), self.global_step, dataformats='CHW')

    def configure_optimizers(self) -> Any:
        params = list(self.parameters())
        opt_net = torch.optim.AdamW(params,
                                    lr=self.lr,
                                    weight_decay=self.weight_decay,
                                    betas=(0.5, 0.9),
                                    )

        opts = [opt_net]

        return opts