import torch
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from losses.loss import Loss

torch.nn.TripletMarginLoss


class TripletMarginLoss(Loss):
    def __init__(self,
                 margin: float = 0.,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.margin = margin

    def forward(self, d0: torch.Tensor, d1: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sign = 1.0 - 2.0 * labels.float()
        loss = (sign * (d0 - d1) + self.margin).clamp(min=0).mean()

        return loss