import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)