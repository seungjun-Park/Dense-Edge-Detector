import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Tuple, Any, Optional

from models.model import Model
from torchvision import models
from collections.abc import Iterable
from modules.downsample.conv import ConvDownSample
from modules.block.res_block import ResidualBlock



class GranularityPredictor(Model):
    def __init__(self,
                 embed_dim: int,
                 num_blocks: Union[int, List[int], Tuple[int]] = 2,
                 drop_path: float = 0.5,
                 activation: str = 'torch.nn.ReLU',
                 use_checkpoint: bool = True,
                 scale_factors: int | List[int] | Tuple[int] = 2,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.encoder = []

        self.encoder.append(nn.Conv2d(4, embed_dim, kernel_size=3, padding=1))

        in_ch = embed_dim

        for i, sf in enumerate(scale_factors):
            for j in range(num_blocks[i] if isinstance(num_blocks, Iterable) else num_blocks):
                self.encoder.append(
                    ResidualBlock(
                        in_channels=in_ch,
                        use_checkpoint=use_checkpoint,
                        activation=activation,
                        drop_path=drop_path,
                    )
                )

            self.encoder.append(
                ConvDownSample(
                    in_channels=in_ch,
                    scale_factor=sf,
                    use_checkpoint=use_checkpoint,
                )
            )

            in_ch = int(in_ch * sf)

        self.encoder = nn.Sequential(*self.encoder)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_ch // 4, 1, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8 * 8, 1)
        )

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.encoder(inputs)
        logits = self.classifier(logits)
        logits = torch.flatten(logits, 1)
        return F.sigmoid(logits)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        images, edges, granularity = batch
        outputs = self(torch.cat([images, edges], dim=1))

        loss, loss_log = self.loss(outputs, granularity, split='train' if self.training else 'valid')

        self.log_dict(loss_log, prog_bar=True)

        return loss



