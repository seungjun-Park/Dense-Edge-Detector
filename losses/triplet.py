import torch
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from losses.loss import Loss


class TripletMarginLoss(Loss):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, d0: torch.Tensor, d1: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
        loss = F.relu(d0 - d1 + margin)

        return loss