import torch
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from losses.loss import Loss


class LMIESLoss(Loss):
    def __init__(self,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)

    def forward(self, feat_imgs: torch.Tensor,
                feat_edges_0: torch.Tensor,
                feat_edges_1: torch.Tensor,
                feat_edges_2: torch.Tensor,
                split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        loss_edges_0_1 = F.triplet_margin_loss(feat_imgs, feat_edges_0, feat_edges_1, margin=0.5) + F.mse_loss(feat_imgs, feat_edges_0)
        loss_edges_1_2 = F.triplet_margin_loss(feat_imgs, feat_edges_1, feat_edges_2, margin=0.5)
        loss_edges_0_2 = F.triplet_margin_loss(feat_imgs, feat_edges_0, feat_edges_2, margin=1.0)

        loss = loss_edges_0_1 + loss_edges_1_2 + loss_edges_0_2

        log_dict = {}

        log_dict.update({f'{split}/total_loss': loss.clone().detach().mean()})
        log_dict.update({f'{split}/loss_edges_0_1': loss_edges_0_1.clone().detach().mean()})
        log_dict.update({f'{split}/loss_edges_1_2': loss_edges_1_2.clone().detach().mean()})
        log_dict.update({f'{split}/loss_edges_0_2': loss_edges_0_2.clone().detach().mean()})


        return loss, log_dict