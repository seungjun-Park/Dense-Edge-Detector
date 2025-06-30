import torch
import torch.nn.functional as F

from piq import SSIMLoss

from typing import Tuple, Dict
from taming.modules.losses import LPIPS
from losses.loss import Loss


class L1(Loss):
    def __init__(self,
                 l1_weight: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.l1_weight = l1_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l1_loss = F.l1_loss(outputs, targets, reduction='mean')

        loss = self.l1_weight * l1_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/l1_loss".format(split): l1_loss.detach().mean(),
               }

        return loss, log