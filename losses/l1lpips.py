import torch
import torch.nn.functional as F

from typing import Tuple, Dict
from taming.modules.losses import LPIPS
from torchmetrics.image import StructuralSimilarityIndexMeasure

from losses.loss import Loss

class L1LPIPS(Loss):
    def __init__(self,
                 lpips_weight: float = 1.0,
                 l1_weight: float = 1.0,
                 content_weight: float = 0.5,
                 ssim_weight: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.perceptual_loss = LPIPS().eval()
        self.vgg16 = self.perceptual_loss.net
        self.lpips_weight = lpips_weight
        self.l1_weight = l1_weight
        self.content_weight = content_weight
        self.ssim_weight = ssim_weight
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, outputs: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l1_loss = F.l1_loss(outputs, targets, reduction='mean')
        ssim_loss =  1 - self.ssim(outputs, targets).mean()

        targets = targets.repeat(1, 3, 1, 1).contiguous()
        outputs = outputs.repeat(1, 3, 1, 1).contiguous()

        lpips_loss = self.perceptual_loss(outputs, targets).mean()
        content_loss = self.perceptual_loss(outputs, inputs).mean()

        loss = self.lpips_weight * lpips_loss + self.l1_weight * l1_loss + self.content_weight * content_loss + self.ssim_weight + ssim_loss

        log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
               "{}/l1_loss".format(split): l1_loss.detach().mean(),
               "{}/lpips_loss".format(split): lpips_loss.detach().mean(),
               "{}/content_loss".format(split): content_loss.detach().mean(),
               }

        return loss, log






