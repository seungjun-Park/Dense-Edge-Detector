import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class MultiLevelTripletLoss(nn.Module):
    def __init__(self,
                 margin_fine=0.05,
                 margin_coarse=0.15,
                 lambda_identity=0.1,
                 start_step: int = 1e3,
                 ):
        super().__init__()
        self.margin_fine = margin_fine
        self.margin_coarse = margin_coarse
        self.lambda_identity = lambda_identity
        self.start_step = start_step

    def forward(self, d_high: torch.Tensor, d_mid: torch.Tensor, d_poor: torch.Tensor, split: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        loss_fine = torch.relu(d_high - d_mid + self.margin_fine).mean()
        log_dict.update({f'{split}/loss_fine': loss_fine.clone().detach().mean()})

        # 3. Coarse-grained Loss (Mid vs Poor)
        # d_poor가 d_mid보다 margin_coarse만큼은 더 커야 함
        loss_coarse = torch.relu(d_mid - d_poor + self.margin_coarse).mean()
        log_dict.update({f'{split}/loss_coarse': loss_coarse.clone().detach().mean()})

        # 4. Identity Loss
        loss_identity = d_high.mean()
        log_dict.update({f'{split}/loss_identity': loss_identity.clone().detach().mean()})

        # 최종 합산
        total_loss = (self.lambda_identity * loss_identity)
        if global_step > self.start_step:
            total_loss += loss_coarse + loss_fine
        log_dict.update({f'{split}/total_loss': total_loss.clone().detach().mean()})

        return total_loss, log_dict