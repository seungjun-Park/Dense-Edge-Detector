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
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        # 导入VGG16模型
        model = models.vgg16_bn(pretrained=True)
        # 加载features部分
        self.features = model.features
        self.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)

        # 加载avgpool层
        self.avgpool = model.avgpool
        # 改变classifier：分类输出层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.save_hyperparameters(ignore='loss_config')

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.features(inputs)
        logits = self.avgpool(logits)
        logits = torch.flatten(logits, 1)
        logits = self.classifier(logits)
        return F.sigmoid(logits)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx) -> Optional[torch.Tensor]:
        images, edges, granularity = batch
        outputs = self(torch.cat([images, edges], dim=1))

        loss, loss_log = self.loss(outputs, granularity, split='train' if self.training else 'valid')

        self.log_dict(loss_log, prog_bar=True)

        return loss



