import torch
import torch.nn.functional as F

from piq import SSIMLoss

from typing import Tuple, Dict, Optional
from taming.modules.losses import LPIPS
from losses.loss import Loss

from models.gnet import GranularityNet


class L1LPIPS(Loss):
    def __init__(self,
                 gnet_ckpt: str,
                 lpips_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 l1_balance_weight: float = 10.,
                 granularity_weight: float = 1.0,
                 threshold: float = 0.4,
                 use_best_quality: bool = False,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.perceptual_loss = LPIPS().eval()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.l1_balance_weight = l1_balance_weight
        self.granularity_weight = granularity_weight
        self.threshold = threshold
        self.use_best_quality = use_best_quality

        if granularity_weight > 0:
            self.gnet = GranularityNet.load_from_checkpoint(gnet_ckpt, strict=False).eval()
            for param in self.gnet.parameters():
                param.requires_grad = False


    def l1_edge_weight(self, edge: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(edge).to(edge.device)
        weight[torch.where(edge < self.threshold)] = self.l1_balance_weight

        return weight

    def forward(self, imgs: torch.Tensor, preds: torch.Tensor, edges: torch.Tensor, labels: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log_dict = {}

        l1_loss = (F.l1_loss(preds, edges, reduction='none') * self.l1_edge_weight(edges)).mean()
        # l1_loss = F.l1_loss(preds, edges).mean()
        log_dict.update({f'{split}/l1_loss': l1_loss.clone().detach().mean()})

        loss = l1_loss * self.l1_weight

        preds = preds.repeat(1, 3, 1, 1).contiguous()
        edges = edges.repeat(1, 3, 1, 1).contiguous()

        if self.lpips_weight > 0.:
            lpips_loss = self.perceptual_loss(edges, preds).mean()
            log_dict.update({f'{split}/lpips_loss': lpips_loss.clone().detach().mean()})
            loss += lpips_loss * self.lpips_weight

        if self.granularity_weight > 0.:
            if self.use_best_quality:
                g_loss = -self.gnet(imgs, preds).mean()
            else:
                g_loss = F.l1_loss(self.gnet(imgs, preds), labels).mean()

            log_dict.update({f'{split}/g_loss': g_loss.clone().detach().mean()})
            loss += g_loss * self.granularity_weight

        log_dict.update({f'{split}/total_loss': loss.clone().detach().mean()})

        return loss, log_dict






