import torch.nn as nn
import re
from typing import Any, Type, Dict, Callable, Tuple
from utils.load_module import load_module
from utils import dictionary_filtering


# LEARNABLE_LAYER_PARAMS
LINEAR_PARAMS = ('in_features', 'out_features','bias', 'device', 'dtype')
CONV_PARAMS = ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'device', 'dtype')

BLOCK_PARAMS = ('in_channels', 'layer', 'out_channels', 'activation', 'norm', 'num_groups', 'dropout', 'drop_path', 'use_checkpoint')
RESIDUAL_BLOCK_PARAMS = (*BLOCK_PARAMS, 'use_conv')
CONV_NEXT_V2_RESIDUAL_PARAMS = (*BLOCK_PARAMS, 'embed_ratio', 'grn')


# NORM_PARAMS
BATCH_NORM_PARAMS = ('num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'device', 'dtype')
LAYER_NORM_PARAMS = ('normalized_shape', 'eps', 'elementwise_affine', 'bias', 'device', 'dtype')
INSTANCE_NORM_PARAMS = ('num_features', 'eps', 'momentum', 'affine', 'track_running_stats', 'device', 'dtype')
GROUPS_NORM_PARAMS = ('num_channels', 'num_groups', 'eps', 'affine', 'device', 'dtype')


# PARAMS_DICTIONARY
PARAMS_GROUPS = {
    'Linear': LINEAR_PARAMS,
    'Conv': CONV_PARAMS,
    'Block': BLOCK_PARAMS,
    'ResidualBlock': RESIDUAL_BLOCK_PARAMS,
    'ConvNextV2ResidualBlock': CONV_NEXT_V2_RESIDUAL_PARAMS,
    'BatchNorm': BATCH_NORM_PARAMS,
}

NAME_PATTERN = r'^[A-Za-z]*'

def get_params_group(module: Type[nn.Module]) -> Tuple[str]:
    module_name = re.match(NAME_PATTERN, repr(module).rsplit('.', 1)[-1]).group(0)

    return PARAMS_GROUPS[module_name]

def get_module_params(module: Type[nn.Module : str]) -> Callable[..., Dict[str, Any]]:
    if isinstance(module, str):
        module = load_module(module)

    def params_dict(*args, **kwargs):
        params = dict()
        params_group = get_params_group(module)

        params.update(dict(zip(params_group, args)))
        params.update(dictionary_filtering(params_group, kwargs))

    return params_dict