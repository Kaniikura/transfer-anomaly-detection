__all__ = [
    'LOSS_REGISTRY',
]

import torch.nn as nn

from ..utils import Null, Registry
from .nf_loss import nf_loss

LOSS_REGISTRY = Registry('loss')
LOSS_REGISTRY.register(nn.MSELoss, name='mse')
LOSS_REGISTRY.register(nf_loss, name='nf_loss')
LOSS_REGISTRY.register(Null, name='none')
