import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class MultiLevelTripletLoss(nn.Module):
    def __init__(self,
                 margin_fine=0.05,
                 margin_coarse=0.15,
                 lambda_identity=0.1
                 ):
        super().__init__()
        self.margin_fine = margin_fine
        self.margin_coarse = margin_coarse
        self.lambda_identity = lambda_identity

    def forward(self, d_high: torch.Tensor, d_mid: torch.Tensor, d_poor: torch.Tensor,
                adaptors_feats: Tuple[torch.Tensor], feats_edges: Tuple[torch.Tensor], split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            d_high, d_mid, d_poor: [B] 형태의 텐서 (Weighted L2 Distance 결과)
        """
        log_dict = {}
        log_dict.update({f'{split}/d_high': d_high.clone().detach().mean()})
        log_dict.update({f'{split}/d_mid': d_mid.clone().detach().mean()})
        log_dict.update({f'{split}/d_poor': d_poor.clone().detach().mean()})

        # 2. Fine-grained Loss (High vs Mid)
        # d_mid가 d_high보다 margin_fine만큼은 더 커야 함
        loss_fine = torch.relu(d_high - d_mid + self.margin_fine)
        log_dict.update({f'{split}/loss_fine': loss_fine.clone().detach().mean()})

        # 3. Coarse-grained Loss (Mid vs Poor)
        # d_poor가 d_mid보다 margin_coarse만큼은 더 커야 함
        loss_coarse = torch.relu(d_mid - d_poor + self.margin_coarse)
        log_dict.update({f'{split}/loss_coarse': loss_coarse.clone().detach().mean()})

        # 4. Identity Loss
        # High 엣지는 원본과 거리가 0에 가까워야 함 (L2 Distance이므로 제곱 불필요, 그 자체로 최소화)
        # Weighted L2는 이미 제곱합의 성격이 있으므로 d_high 자체를 줄이거나 d_high^2를 줄임
        loss_identity = 0
        for adaptor_feats, feat_edges in zip(adaptors_feats, feats_edges):
            loss_identity += F.mse_loss(adaptor_feats, feat_edges)

        log_dict.update({f'{split}/loss_identity': loss_identity.clone().detach().mean()})

        # 최종 합산
        total_loss = torch.mean(loss_fine + loss_coarse) + (self.lambda_identity * loss_identity)
        log_dict.update({f'{split}/total_loss': total_loss.clone().detach().mean()})

        return total_loss, log_dict