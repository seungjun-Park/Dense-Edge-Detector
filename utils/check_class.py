import torch.nn as nn

from typing import Any, Type, Dict, Callable
from utils.load_module import load_module

BATCH_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
INSTANCE_NORM_TYPES = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
LAYER_NORM_TYPES = (nn.LayerNorm,)
GROUP_NORM_TYPES = (nn.GroupNorm,)
NORM_TYPES = (*BATCH_NORM_TYPES, *INSTANCE_NORM_TYPES, *LAYER_NORM_TYPES)

LINEAR_TYPES = (nn.Linear, )

CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
CONV_TRANSPOSE_TYPES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def is_batch_norm(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, BATCH_NORM_TYPES)

def is_layer_norm(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, LAYER_NORM_TYPES)

def is_instance_norm(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, INSTANCE_NORM_TYPES)

def is_group_norm(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, GROUP_NORM_TYPES)

def is_norm(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, NORM_TYPES)

def is_linear(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, LINEAR_TYPES)

def is_conv(module: Type[nn.Module] | str) -> bool:
    return issubclass(module, CONV_TYPES)