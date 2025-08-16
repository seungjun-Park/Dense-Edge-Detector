import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, List

class Dist2LogitLayer(nn.Module):
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        self.dists2dist = nn.Sequential(
            nn.Conv2d(6, chn_mid, 1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(0.2, True)
        )

        layers = [nn.Conv2d(chn_mid * 2, chn_mid, 1, stride=1, padding=0, bias=True), ]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if use_sigmoid:
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self, d0: torch.Tensor, d1: torch.Tensor, eps=1e-6):
        d0_align, d0_raw, d0_shift = d0.chunk(3, dim=1)
        d1_align, d1_raw, d1_shift = d1.chunk(3, dim=1)

        d0 = self.dists2dist(torch.cat([d0_align, d0_raw, d0_align - d0_raw, d0_align / (d0_raw + eps), d0_raw / (d0_align + eps), d0_shift], dim=1))
        d1 = self.dists2dist(torch.cat([d1_align, d1_raw, d1_align - d1_raw, d1_align / (d1_raw + eps), d1_raw / (d1_align + eps), d1_shift], dim=1))

        return self.model.forward(torch.cat([d0, d1],dim=1)).reshape(-1, 1)


class PredictionLoss(nn.Module):
    def __init__(self,
                 abs_weight: float = 1.0,
                 bce_weight: float = 1.0,
                 *args,
                 **kwargs
                 ):

        super().__init__(*args, **kwargs)
        self.abs_weight = abs_weight
        self.bce_weight = bce_weight
        self.net = Dist2LogitLayer()
        self.bce_loss = nn.BCELoss(reduction='mean')

    def _check_pairs(self, g: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
        """
        g: [B, 1] 또는 [B]  (값은 [0,1])
        thr: abs(g_i - g_j) > thr 인 (i<j) 쌍만 반환
        return: (i_idx, j_idx) 각각 [N_pairs]
        """
        # [B,1] -> [B]
        if g.dim() == 2 and g.size(-1) == 1:
            g = g.squeeze(-1)
        if g.dim() != 1:
            raise ValueError("g must be shape [B] or [B,1].")

        B = g.numel()
        # 쌍별 차이
        G = g[:, None] - g[None, :]  # [B,B]
        # i<j 상삼각 + 임계치 조건
        triu = torch.triu(torch.ones(B, B, dtype=torch.bool, device=g.device), diagonal=1)
        mask = triu & (G.abs() > threshold)
        i_idx, j_idx = mask.nonzero(as_tuple=True)
        labels = G.sign()[mask].clamp(0, 1).unsqueeze(-1)
        i_idx, j_idx = i_idx.detach().cpu().tolist(), j_idx.detach().cpu().tolist()
        if len(i_idx) > 0:
            return zip(i_idx, j_idx), labels
        return None

    def forward(self, g_hat: torch.Tensor, g: torch.Tensor, d_align: torch.Tensor, d_raw: torch.Tensor, d_shift: torch.Tensor, split: str, threshold: float = 0.04) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        log = dict()

        loss_abs = F.huber_loss(g_hat, g, reduction='mean', delta=0.5)
        log.update({f'{split}/loss_abs': loss_abs.detach().clone()})

        loss = self.abs_weight * loss_abs

        b, _ = g.shape

        if b > 1:
            pair_idx, rank_labels = self._check_pairs(g, threshold)
            if pair_idx is not None:
                d_i, d_j = [], []
                for i, j in pair_idx:
                    d_i.append(torch.cat([d_align[i], d_raw[i], d_shift[i]], dim=0))
                    d_j.append(torch.cat([d_align[j], d_raw[j], d_shift[j]], dim=0))

                d_i = torch.stack(d_i, dim=0)
                d_j = torch.stack(d_j, dim=0)

                loss_bce = self.bce_loss(self.net(d_i, d_j), rank_labels)
                log.update({f'{split}/loss_bce': loss_bce.detach().clone()})

                loss += self.bce_weight * loss_bce

        log.update({f'{split}/loss_total': loss.detach().clone()})

        return loss, log