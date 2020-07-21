import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from cvpods.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")
LR_SCHEDULERS = Registry("lr_schedulers")

for attr in dir(torch.optim):
    optim = getattr(torch.optim, attr)
    try:
        if issubclass(optim, Optimizer):
            OPTIMIZERS.register(optim)
    except:
        continue

for attr in dir(torch.optim.lr_scheduler):
    lrs = getattr(torch.optim.lr_scheduler, attr)
    try:
        if issubclass(lrs, _LRScheduler):
            LR_SCHEDULERS.register(lrs)
    except:
        continue
