import torch
import torch.nn.functional as F

from piq import SSIMLoss

from typing import Tuple, Dict
from taming.modules.losses import LPIPS
from losses.loss import Loss

from models.discriminator import Discriminator


class L1LPIPS(Loss):
    def __init__(self,
                 granularity_ckpt: str = './checkpoints/granularity_prediction/hybrid/best.ckpt',
                 lpips_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 content_weight: float = 0.5,
                 ssim_weight: float = 1.0,
                 granularity_weight: float = 0.,
                 start_step: int = 0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.perceptual_loss = LPIPS().eval()
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.content_weight = content_weight
        self.ssim_weight = ssim_weight
        self.ssim_loss = SSIMLoss(data_range=1.001)
        self.granularity_weight = granularity_weight
        self.start_step = start_step

        if self.granularity_weight > 0.:
            self.granularity = Discriminator.load_from_checkpoint(granularity_ckpt).eval()
            for p in self.granularity.parameters():
                p.requires_grad = False

    def l1_edge_weight(self, edge: torch.Tensor) -> torch.Tensor:
        weight = torch.ones_like(edge).to(edge.device)
        weight[torch.where(edge < 0.8)] += 0.2

        return weight


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor,
                granularity: torch.Tensor, global_step: int,  split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l1_loss = (F.l1_loss(outputs, targets, reduction='none') * self.l1_edge_weight(targets)).mean()
        ssim_loss = self.ssim_loss(outputs, targets).mean()
        granularity_loss = F.l1_loss(granularity, self.granularity(inputs, outputs))

        targets = targets.repeat(1, 3, 1, 1).contiguous()
        outputs = outputs.repeat(1, 3, 1, 1).contiguous()

        lpips_loss = self.perceptual_loss(outputs, targets).mean()
        content_loss = self.perceptual_loss(outputs, inputs).mean()

        loss = (self.lpips_weight * lpips_loss +
                self.l1_weight * l1_loss +
                self.content_weight * content_loss +
                self.ssim_weight * ssim_loss)

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/l1_loss".format(split): l1_loss.detach().mean(),
               "{}/lpips_loss".format(split): lpips_loss.detach().mean(),
               "{}/content_loss".format(split): content_loss.detach().mean(),
               "{}/ssim_loss".format(split): ssim_loss.detach().mean(),
               "{}/granularity_loss".format(split): granularity_loss.detach().mean(),
               }

        if global_step > self.start_step:
            loss += self.granularity_weight * granularity_loss

            log.update({f"{split}/granularity_loss": granularity_loss.detach().mean()})

        return loss, log






