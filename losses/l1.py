import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict



class L1(nn.Module):
    def __init__(self,
                 l1_weight: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.l1_weight = l1_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l1_loss = F.l1_loss(outputs, targets, reduction='mean')

        loss = self.l1_weight * l1_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/l1_loss".format(split): l1_loss.detach().mean(),
               }

        return loss, log