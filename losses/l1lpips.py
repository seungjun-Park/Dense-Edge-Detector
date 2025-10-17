import torch
import torch.nn.functional as F

from piq import SSIMLoss

from typing import Tuple, Dict, Optional
from taming.modules.losses import LPIPS
from losses.loss import Loss

from models.discriminator import Discriminator
from models.lpieps import LPIEPS


class L1LPIPS(Loss):
    def __init__(self,
                 lpieps_ckpt: str = './checkpoints/lpieps/vgg/best.ckpt',
                 lpips_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 lpieps_start_step: int = 1e-3,
                 ema_decay: float = 0.99,
                 ema_lpieps_lv0: float = 0.4188,
                 ema_lpieps_lv2: float = 0.9916,
                 lpieps_lv0_weight: float = 1.0,
                 lpieps_lv1_weight: float = 1.0,
                 lpieps_lv2_weight: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.perceptual_loss = LPIPS().eval()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.lpieps_start_step = lpieps_start_step

        self.ema_decay = ema_decay

        self.register_buffer('ema_lpieps_lv0', torch.tensor(ema_lpieps_lv0))
        self.register_buffer('ema_lpieps_lv2', torch.tensor(ema_lpieps_lv2))

        self.lpieps = LPIEPS.load_from_checkpoint(f'{lpieps_ckpt}', strict=False).eval()
        self.lpieps_lv0_weight = lpieps_lv0_weight
        self.lpieps_lv1_weight = lpieps_lv1_weight
        self.lpieps_lv2_weight = lpieps_lv2_weight

        for param in self.lpieps.parameters():
            param.requires_grad = False

    def l1_edge_weight(self, edge: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(edge).to(edge.device)
        weight[torch.where(edge < 0.8)] += 0.2

        return weight

    def get_l1lpips_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l1_loss = (F.l1_loss(outputs, targets, reduction='none') * self.l1_edge_weight(targets)).mean()

        loss = l1_loss * self.l1_weight

        targets = targets.repeat(1, 3, 1, 1).contiguous()
        outputs = outputs.repeat(1, 3, 1, 1).contiguous()

        if self.lpips_weight > 0.:
            lpips_loss = self.perceptual_loss(targets, outputs).mean()
            loss += lpips_loss * self.lpips_weight

        return loss

    def forward(self, imgs: torch.Tensor,
                edges0: torch.Tensor, edges1: torch.Tensor, edges2: torch.Tensor,
                preds0: torch.Tensor, preds1: torch.Tensor, preds2: torch.Tensor,
                global_step: int,  split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        log_dict = {}

        loss_lv_0 = self.get_l1lpips_loss(preds0, edges0)
        loss_lv_1 = self.get_l1lpips_loss(preds1, edges1)
        loss_lv_2 = self.get_l1lpips_loss(preds2, edges2)

        if global_step > self.lpieps_start_step:
            lpieps_lv0 = torch.clamp(self.lpieps(imgs, preds0), max=1.2)
            lpieps_lv1 = torch.clamp(self.lpieps(imgs, preds1), max=1.2)
            lpieps_lv2 = torch.clamp(self.lpieps(imgs, preds2), max=1.2)
            with torch.no_grad():
                self.ema_lpieps_lv0 = self.ema_decay * self.ema_lpieps_lv0 + (1 - self.ema_decay) * lpieps_lv0
                self.ema_lpieps_lv2 = self.ema_decay * self.ema_lpieps_lv2 + (1 - self.ema_decay) * lpieps_lv2

            margin = (self.ema_lpieps_lv0 - self.ema_lpieps_lv2) / 2.0

            loss_dist_from_lv0 = torch.abs((self.ema_lpieps_lv0 - lpieps_lv1) - margin)
            loss_dist_from_lv2 = torch.abs((lpieps_lv1 - self.ema_lpieps_lv2) - margin)
            consistency_loss = loss_dist_from_lv0 + loss_dist_from_lv2

            log_dict.update({f'{split}/lpieps_lv0': lpieps_lv0.clone().detach().mean()})
            log_dict.update({f'{split}/lpieps_lv1': lpieps_lv1.clone().detach().mean()})
            log_dict.update({f'{split}/lpieps_lv2': lpieps_lv2.clone().detach().mean()})

            loss_lv_0 += lpieps_lv0.mean() * self.lpieps_lv0_weight
            loss_lv_1 += consistency_loss.mean() * self.lpieps_lv1_weight
            loss_lv_2 += -lpieps_lv2.mean() * self.lpieps_lv2_weight

        loss = loss_lv_0 + loss_lv_1 + loss_lv_2

        log_dict.update({f'{split}/total_loss': loss.clone().detach().mean()})
        log_dict.update({f'{split}/loss_lv_0': loss_lv_0.clone().detach().mean()})
        log_dict.update({f'{split}/loss_lv_1': loss_lv_1.clone().detach().mean()})
        log_dict.update({f'{split}/loss_lv_2': loss_lv_2.clone().detach().mean()})


        return loss, log_dict






