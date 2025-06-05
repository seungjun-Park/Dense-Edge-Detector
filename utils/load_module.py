import torch.nn as nn
import importlib

from typing import Any, Type, Dict, Callable


def load_module(name: str) -> Any:
    if name is None:
        return nn.Identity()

    lib, cls = name.rsplit(".", 1)
    module = getattr(importlib.import_module(lib, package=None), cls)

    return module