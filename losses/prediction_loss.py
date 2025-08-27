import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, List

class Dist2LogitLayer(nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if use_sigmoid:
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self, d0: torch.Tensor, d1: torch.Tensor, eps=1e-6):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))


def adaptive_margin_ranking_loss(x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, margins: torch.Tensor, reduction: str = 'mean'):
    reduction = reduction.lower()
    assert reduction in ['none', 'sum', 'mean']
    loss = F.relu(-y * (x0 - x1) + margins)
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()

    return loss


class PredictionLoss(nn.Module):
    def __init__(self,
                 weight: float = 10.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.weight = weight

    def forward(self, d0: torch.Tensor, d1: torch.Tensor, margins: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log = dict()

        y = margins.sign()
        margins = margins.abs()

        loss = self.weight * adaptive_margin_ranking_loss(d0, d1, y, margins=margins)

        log.update({f'{split}/loss': loss.detach().cpu().clone()})

        return loss, log